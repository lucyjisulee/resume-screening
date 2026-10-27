import io
import json
import os
import re
import time
import google.generativeai as genai
import pandas as pd
from pypdf import PdfReader
import streamlit as st

# --- 1. 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 채용 서류 스크리닝 시스템", page_icon="📋", layout="wide"
)

JOBS_FILE = "jobs_data.json"


# --- 2. 데이터 영구 저장/불러오기 ---
def load_jobs():
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_jobs(jobs):
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)


# --- 3. 개인정보 자동 마스킹 (보안 필터) ---
def mask_personal_info(text: str) -> str:
    if not text:
        return ""
    # 주민등록번호 패턴 마스킹 (950101-1234567, 9501011234567)
    text = re.sub(r"\b(\d{6})[- ]?([1-4]\d{6})\b", r"\1-*******", text)
    # 휴대전화 및 일반 전화번호 마스킹 (010-1234-5678, 02-123-4567 등)
    text = re.sub(
        r"\b(01[016789]|02|0[3-9]\d{1})[- ]?(\d{3,4})[- ]?(\d{4})\b",
        r"\1-****-\3",
        text,
    )
    # 이메일 주소 마스킹 (user@example.com -> u***@example.com)
    text = re.sub(
        r"\b([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b",
        r"\1***@\2",
        text,
    )
    return text


# --- 4. 사람인 통합 PDF 자동 분할 파서 ---
def split_candidates_from_pdf(uploaded_file):
    """사람인 일괄 인쇄 PDF에서 지원자 단위로 페이지를 인식해 텍스트를 분할합니다."""
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)

    if total_pages == 0:
        return []

    # 전체 페이지 텍스트 추출
    pages_text = []
    for p in reader.pages:
        extracted = p.extract_text() or ""
        pages_text.append(extracted)

    # 사람인 인쇄 헤더 마커
    split_marker = "지원자 이력서 인쇄하기"
    marker_indices = [
        idx for idx, txt in enumerate(pages_text) if split_marker in txt
    ]

    # 마커 감지 기준 지원자 분할점 계산
    split_indices = []
    # 마커가 2회 이상 발견되고, 매 페이지마다 찍히는 브라우저 머리글이 아닌 경우
    if len(marker_indices) >= 2 and len(marker_indices) < total_pages:
        split_indices = marker_indices
    else:
        # 보조 분할 로직: "기본정보/인적사항"과 "학력/경력"이 동시에 등장하는 첫 장 감지
        for idx, txt in enumerate(pages_text):
            if idx == 0:
                split_indices.append(idx)
            elif any(k in txt for k in ["기본정보", "인적사항", "입사지원서"]) and any(
                k in txt for k in ["학력사항", "경력사항", "생년월일", "희망연봉"]
            ):
                split_indices.append(idx)

    if not split_indices:
        split_indices = [0]

    # 지원자별로 페이지 구간 묶기
    candidate_chunks = []
    for i in range(len(split_indices)):
        start_p = split_indices[i]
        end_p = (
            split_indices[i + 1] if i + 1 < len(split_indices) else total_pages
        )
        chunk_text = "\n".join(pages_text[start_p:end_p]).strip()

        if chunk_text:
            candidate_chunks.append(
                {
                    "candidate_seq": i + 1,
                    "page_range": f"p.{start_p + 1} ~ p.{end_p}",
                    "text": chunk_text,
                    "filename": uploaded_file.name,
                }
            )

    return candidate_chunks


# --- 5. 안전한 Gemini API 호출 엔진 (모델 폴백 + 속도 제한 대응) ---
def call_gemini_safe(prompt: str, max_retries: int = 3) -> str:
    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    last_error = None

    for attempt in range(max_retries):
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json"
                    },
                )
                return response.text
            except Exception as e:
                err_msg = str(e).lower()
                last_error = e
                # 지원 종료 모델일 경우 다음 모델 즉시 시도
                if "404" in err_msg or "not found" in err_msg:
                    continue
                # API 호출 한도(429) 초과 시 잠시 대기 후 재시도
                elif (
                    "429" in err_msg
                    or "quota" in err_msg
                    or "resourceexhausted" in err_msg
                ):
                    time.sleep(3 * (attempt + 1))
                    break
                else:
                    time.sleep(1)
                    break

    raise Exception(f"AI 응답 생성 실패: {str(last_error)}")


# --- 6. 사이드바 보안 인증 ---
with st.sidebar:
    st.header("🔒 시스템 접근 인증")
    env_password = st.secrets.get("TEAM_PASSWORD", "")
    env_api_key = st.secrets.get("GEMINI_API_KEY", "")

    if env_password:
        user_password = st.text_input(
            "팀 비밀번호", type="password", value=st.session_state.get("auth_pw", "")
        )
        if user_password != env_password:
            st.error("비밀번호가 올바르지 않습니다.")
            st.stop()
        st.session_state["auth_pw"] = user_password
        st.success("인증 완료")
    else:
        st.info("💡 Secrets에 TEAM_PASSWORD를 설정하면 암호가 활성화됩니다.")

    st.markdown("---")
    st.subheader("🔑 AI 엔진 설정")
    api_key = env_api_key or st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
    )

    if api_key:
        st.session_state["api_key"] = api_key
        genai.configure(api_key=api_key)
    else:
        st.warning("API 키를 입력해 주세요.")
        st.stop()

# --- 7. 메인 화면 ---
tab_screening, tab_job_manager = st.tabs(
    ["📄 이력서 대량 스크리닝", "⚙️ 공고 관리 (상시 업데이트)"]
)
jobs_dict = load_jobs()

# ==========================================
# 탭 1: 이력서 대량 스크리닝
# ==========================================
with tab_screening:
    st.title("📄 지원자 이력서 자동 스크리닝")
    st.caption(
        "사람인 통합 PDF(20~80명 분량)를 통째로 업로드해도 지원자별로 자동 쪼개어 정밀 평가합니다."
    )

    if not jobs_dict:
        st.warning(
            "등록된 채용 공고가 없습니다. 먼저 **'⚙️ 공고 관리'** 탭에서 공고를 등록해 주세요."
        )
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_job_title = st.selectbox(
                "평가할 공고 선택", list(jobs_dict.keys())
            )
            job_info = jobs_dict[selected_job_title]

            with st.expander("📌 선택된 공고 기준 확인", expanded=False):
                st.markdown(f"**담당 업무:**\n{job_info.get('duties', '-')}")
                st.markdown(f"**필수 요건:**\n{job_info.get('required', '-')}")
                st.markdown(f"**우대 사항:**\n{job_info.get('preferred', '-')}")

        with col2:
            uploaded_files = st.file_uploader(
                "사람인 통합 이력서 PDF 파일 업로드 (복수 업로드 가능)",
                type=["pdf"],
                accept_multiple_files=True,
            )

        start_btn = st.button("🚀 서류 스크리닝 시작", type="primary")

        if start_btn and uploaded_files:
            # 1단계: 모든 파일에서 지원자 단위 청크 분할 수집
            all_candidates = []
            with st.spinner("PDF 문서에서 개별 지원자 영역을 분할하는 중..."):
                for up_file in uploaded_files:
                    chunks = split_candidates_from_pdf(up_file)
                    all_candidates.extend(chunks)

            total_cands = len(all_candidates)
            if total_cands == 0:
                st.error(
                    "이력서 텍스트를 읽을 수 없습니다. 정상적인 PDF 파일인지 확인해 주세요."
                )
            else:
                st.info(
                    f"💡 총 **{len(uploaded_files)}개**의 파일에서 **{total_cands}명**의 지원자를 성공적으로 감지했습니다. 분석을 시작합니다."
                )

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, cand in enumerate(all_candidates):
                    # 진행 상황 표시
                    status_text.text(
                        f"[{idx+1}/{total_cands}] 지원자 분석 중 ({cand['page_range']})..."
                    )

                    # 개인정보 마스킹 처리
                    masked_text = mask_personal_info(cand["text"])

                    eval_prompt = f"""
당신은 기업의 전문 인사담당자(채용 심사관)입니다.
아래 채용 공고 기준과 지원자 이력서를 엄격히 대조 평가하여 JSON 형식으로만 응답하세요.

[채용 공고 기준]
- 공고명: {selected_job_title}
- 담당 업무: {job_info.get('duties')}
- 필수 요건: {job_info.get('required')}
- 우대 요건: {job_info.get('preferred')}

[평가 및 등급 부여 규칙]
1. 등급 체계:
   - S: 필수 요건 완벽 충족 + 우대 사항 다수 충족 + 뛰어난 직무 핏
   - A: 필수 요건 충족 + 우대 사항 일부 충족
   - B: 필수 요건 충족 + 우대 사항 미미
   - C: 필수 요건 충족이 다소 애매하거나 직무 연관성 낮음
   - 탈락: 필수 요건 명백히 미충족
2. 학점(GPA) 예외 규칙:
   - 학점 3.0 이상을 권장하나, 이력서에 학점이 전혀 적혀있지 않은 경우 절대 탈락(Fail) 사유로 삼지 마세요.
   - 학점이 기재되지 않았다면 '메모_확인필요'에 반드시 '이력서 확인 필요(학점 미기재)'라고 적고, 경력/프로젝트 등 다른 필수 요건을 중심으로 등급을 산정하세요.

[지원자 이력서 원문 (개인식별정보 마스킹됨)]
{masked_text[:12000]}

[출력 형식]
반드시 유효한 JSON 형식만 출력하세요.
{{
  "지원자명": "지원자 이름 (이력서 상단에서 추출, 식별 불가 시 '지원자 {cand['candidate_seq']}번')",
  "평가등급": "S, A, B, C, 탈락 중 택 1",
  "필수요건_충족여부": "충족 / 미충족 / 보완필요 중 택 1",
  "우대요건_해당사항": "충족하는 우대 요건 요약",
  "종합_평가사유": "서류 등급 부여 핵심 사유 (2~3문장)",
  "메모_확인필요": "학점 미기재 확인 필요 여부 또는 특이사항 (없으면 '해당 없음')"
}}
"""
                    try:
                        raw_json = call_gemini_safe(eval_prompt)
                        parsed = json.loads(raw_json)
                        parsed["페이지"] = cand["page_range"]
                        parsed["파일명"] = cand["filename"]
                        results.append(parsed)
                    except Exception as e:
                        results.append(
                            {
                                "지원자명": f"지원자 {cand['candidate_seq']}번",
                                "평가등급": "오류",
                                "필수요건_충족여부": "오류",
                                "우대요건_해당사항": "-",
                                "종합_평가사유": f"분석 오류 발생: {str(e)}",
                                "메모_확인필요": "수동 확인 필요",
                                "페이지": cand["page_range"],
                                "파일명": cand["filename"],
                            }
                        )

                    progress_bar.progress((idx + 1) / total_cands)
                    # API 연속 호출 과부하 방지를 위한 미세 딜레이
                    time.sleep(1.2)

                status_text.text(
                    f"✅ 총 {total_cands}명의 서류 스크리닝이 완료되었습니다!"
                )

                df = pd.DataFrame(results)
                cols_order = [
                    "지원자명",
                    "평가등급",
                    "필수요건_충족여부",
                    "우대요건_해당사항",
                    "종합_평가사유",
                    "메모_확인필요",
                    "페이지",
                    "파일명",
                ]
                df = df[[c for c in cols_order if c in df.columns]]

                st.subheader("📊 스크리닝 종합 결과")
                st.dataframe(df, use_container_width=True)

                # 엑셀 다운로드 파일 생성
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="스크리닝결과")
                excel_data = excel_buffer.getvalue()

                st.download_button(
                    label="📥 엑셀(.xlsx) 다운로드",
                    data=excel_data,
                    file_name=f"{selected_job_title}_서류스크리닝결과.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

# ==========================================
# 탭 2: 공고 관리 (매주 유동적 추가/삭제/AI 추출)
# ==========================================
with tab_job_manager:
    st.title("⚙️ 채용 공고 관리")
    st.caption("사람인 공고를 복사해 넣으면 AI가 기준을 추출하여 등록합니다.")

    st.subheader("➕ 새 공고 등록")
    raw_job_text = st.text_area(
        "사람인 공고 본문 붙여넣기 (담당업무/필수/우대 위주)",
        height=180,
        placeholder="채용 공고 본문을 복사하여 여기에 붙여넣으세요...",
    )

    if st.button("🤖 AI로 공고 기준 자동 추출 및 등록"):
        if not raw_job_text.strip():
            st.error("공고 텍스트를 입력해 주세요.")
        else:
            with st.spinner("AI가 공고 요건을 파악하고 있습니다..."):
                extract_prompt = f"""
채용 사이트에서 복사한 원문 텍스트입니다.
공고명, 주요 업무, 지원 필수 자격, 우대 사항을 추출하여 순수한 JSON 형식으로 반환하세요.

[원문 텍스트]
{raw_job_text[:10000]}

[출력 형식]
{{
  "title": "공고명 (예: 퍼포먼스 마케터 3년 이상)",
  "duties": "담당 업무 항목 요약",
  "required": "필수 자격 요건 요약",
  "preferred": "우대 사항 요약"
}}
"""
                try:
                    res_json = call_gemini_safe(extract_prompt)
                    parsed_job = json.loads(res_json)
                    job_title = parsed_job.get("title", "새 채용공고")

                    jobs_dict[job_title] = {
                        "duties": parsed_job.get("duties", ""),
                        "required": parsed_job.get("required", ""),
                        "preferred": parsed_job.get("preferred", ""),
                    }
                    save_jobs(jobs_dict)
                    st.success(f"'{job_title}' 공고가 등록되었습니다!")
                    st.rerun()
                except Exception as e:
                    st.error(f"공고 추출 실패: {str(e)}")

    st.markdown("---")
    st.subheader(f"📋 등록된 공고 목록 ({len(jobs_dict)}개)")

    if jobs_dict:
        for title, info in list(jobs_dict.items()):
            with st.expander(f"📌 {title}"):
                st.text_area(
                    "담당 업무",
                    value=info.get("duties", ""),
                    key=f"duties_{title}",
                    disabled=True,
                )
                st.text_area(
                    "필수 요건",
                    value=info.get("required", ""),
                    key=f"req_{title}",
                    disabled=True,
                )
                st.text_area(
                    "우대 사항",
                    value=info.get("preferred", ""),
                    key=f"pref_{title}",
                    disabled=True,
                )

                if st.button("🗑️ 이 공고 삭제", key=f"del_{title}"):
                    del jobs_dict[title]
                    save_jobs(jobs_dict)
                    st.warning(f"'{title}' 공고가 삭제되었습니다.")
                    st.rerun()
    else:
        st.info("현재 등록된 공고가 없습니다.")
