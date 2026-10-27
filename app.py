import io
import json
import os
import re
import time
import google.generativeai as genai
import pandas as pd
from pypdf import PdfReader
import streamlit as st

# --- 기본 페이지 설정 ---
st.set_page_config(
    page_title="AI 채용 서류 스크리닝 시스템", page_icon="📋", layout="wide"
)

JOBS_FILE = "jobs_data.json"


# --- 공고 저장/불러오기 ---
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


# --- 개인정보 자동 마스킹 (휴대폰, 유선번호, 이메일, 주민번호, 상세주소) ---
def mask_personal_info(text: str) -> str:
    if not text:
        return ""
    # 1. 주민등록번호 패턴
    text = re.sub(r"\b(\d{6})[- ]?([1-4]\d{6})\b", r"\1-*******", text)
    # 2. 휴대폰 및 전화번호 마스킹 (010-4170-8558 -> 010-****-8558)
    text = re.sub(
        r"\b(01[016789]|02|0[3-9]\d{1})[- ]?(\d{3,4})[- ]?(\d{4})\b",
        r"\1-****-\3",
        text,
    )
    # 3. 이메일 주소 마스킹 (jsj970408@naver.com -> j***@naver.com)
    text = re.sub(
        r"\b([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b",
        r"\1***@\2",
        text,
    )
    # 4. 상세 주소 마스킹 (시/구 까지만 남기고 뒷부분 마스킹)
    text = re.sub(
        r"(\([0-9]{5}\)\s*[가-힣]+(?:\s+[가-힣]+)?)\s+[가-힣0-9\s-]+",
        r"\1 [상세주소 마스킹]",
        text,
    )
    return text


# --- 사람인 일괄 인쇄 PDF 정밀 분할기 ---
def split_candidates_from_pdf(uploaded_file):
    """사람인 이력서 상단의 '입사지원일 :' 및 '지원분야 :' 표식을 감지하여 지원자별로 정확히 분할합니다."""
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)

    if total_pages == 0:
        return []

    pages_text = [p.extract_text() or "" for p in reader.pages]
    split_indices = []

    # 사람인 각 지원자의 첫 페이지에만 등장하는 고유 표식 탐색
    for idx, txt in enumerate(pages_text):
        if idx == 0:
            split_indices.append(0)
            continue

        has_apply_date = bool(re.search(r"입사지원일\s*:", txt))
        has_apply_field = bool(re.search(r"지원분야\s*:", txt))
        has_profile_meta = bool(
            re.search(r"(남|여),\s*(19|20)\d{2}\s*\(\d{1,2}세\)", txt)
        )

        # 지원분야와 입사지원일이 동시에 있거나, 인적사항 메타 정보가 매칭되는 경우 새 지원자 시작
        if (has_apply_date and has_apply_field) or (
            (has_apply_date or has_apply_field) and has_profile_meta
        ):
            split_indices.append(idx)

    split_indices = sorted(list(set(split_indices)))

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


# --- Gemini API 안전 호출 (최신 gemini-3.8-flash 우선 적용) ---
def call_gemini_safe(prompt: str, max_retries: int = 3) -> str:
    # 에러 방지를 위해 3.8-flash 우선, 하위 호환 폴백
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
                if "404" in err_msg or "not found" in err_msg:
                    continue  # 지원 중단 모델이면 다음 모델 시도
                elif (
                    "429" in err_msg
                    or "quota" in err_msg
                    or "resourceexhausted" in err_msg
                ):
                    time.sleep(3 * (attempt + 1))  # 초당 호출 한도 초과 시 대기
                    break
                else:
                    time.sleep(1)
                    break

    raise Exception(f"AI 응답 생성 실패: {str(last_error)}")


# --- 사이드바 인증 ---
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
        st.warning("API 키를 등록해 주세요.")
        st.stop()

# --- 메인 화면 탭 구성 ---
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
        "사람인 일괄 인쇄 PDF(20~80명)를 올리면 지원자별로 자동 분할하여 개별 심사합니다."
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
                "사람인 통합 이력서 PDF 파일 업로드",
                type=["pdf"],
                accept_multiple_files=True,
            )

        start_btn = st.button("🚀 서류 스크리닝 시작", type="primary")

        if start_btn and uploaded_files:
            all_candidates = []
            with st.spinner("이력서 PDF에서 개별 지원자 영역을 분할하는 중..."):
                for up_file in uploaded_files:
                    chunks = split_candidates_from_pdf(up_file)
                    all_candidates.extend(chunks)

            total_cands = len(all_candidates)
            if total_cands == 0:
                st.error("이력서 텍스트를 인식할 수 없습니다.")
            else:
                st.success(
                    f"💡 총 **{len(uploaded_files)}개**의 파일에서 **{total_cands}명**의 지원자를 감지했습니다. 심사를 시작합니다!"
                )

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, cand in enumerate(all_candidates):
                    status_text.text(
                        f"[{idx+1}/{total_cands}] 지원자 분석 중 ({cand['page_range']})..."
                    )

                    # 보안을 위한 개인정보 자동 마스킹
                    masked_text = mask_personal_info(cand["text"])

                    eval_prompt = f"""
당신은 기업의 전문 채용 심사관입니다.
아래 채용 공고 기준과 지원자 이력서를 비교 평가하여 JSON 형식으로만 응답하세요.

[채용 공고 기준]
- 공고명: {selected_job_title}
- 담당 업무: {job_info.get('duties')}
- 필수 요건: {job_info.get('required')}
- 우대 요건: {job_info.get('preferred')}

[평가 및 등급 부여 규칙]
1. 등급 체계:
   - S: 필수 요건 완벽 충족 + 우대 사항 다수 충족 + 뛰어난 직무 역량
   - A: 필수 요건 충족 + 우대 사항 일부 충족
   - B: 필수 요건 충족 + 우대 사항 미미
   - C: 필수 요건 충족 애매하거나 직무 핏 낮음
   - 탈락: 필수 요건 명백히 미충족
2. 학점(GPA) 규칙:
   - 학점 3.0 이상 권장이나, 이력서에 학점이 아예 적혀있지 않은 경우 절대 탈락(Fail)시키지 마세요.
   - 학점 미기재 시 '메모_확인필요'에 '이력서 확인 필요(학점 미기재)'라고 적고, 경력/자격 등 다른 필수 요건을 중심으로 등급을 산정하세요.

[지원자 이력서 원문 (개인정보 마스킹됨)]
{masked_text[:12000]}

[출력 형식 (순수 JSON)]
{{
  "지원자명": "이력서 상단의 지원자 성명 (추정 불가 시 '지원자 {cand['candidate_seq']}번')",
  "평가등급": "S, A, B, C, 탈락 중 택 1",
  "필수요건_충족여부": "충족 / 미충족 / 보완필요 중 택 1",
  "우대요건_해당사항": "충족하는 우대 요건 요약",
  "종합_평가사유": "서류 합격/탈락 및 등급 부여 사유 (2~3문장)",
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
                                "종합_평가사유": f"분석 오류: {str(e)}",
                                "메모_확인필요": "수동 확인 필요",
                                "페이지": cand["page_range"],
                                "파일명": cand["filename"],
                            }
                        )

                    progress_bar.progress((idx + 1) / total_cands)
                    time.sleep(1.0)  # 안정적인 연속 호출 간격

                status_text.text(
                    f"✅ 총 {total_cands}명의 스크리닝이 모두 완료되었습니다!"
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

                # 엑셀 다운로드
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
        "사람인 공고 본문 붙여넣기 (담당업무/자격요건/우대사항 위주)",
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
  "title": "공고명 (예: 안전보건 관리자)",
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
