import io
import json
import os
import re
import time
import google.generativeai as genai
import pandas as pd
from pypdf import PdfReader
import streamlit as st

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title="AI 채용 서류 스크리닝 시스템", page_icon="📋", layout="wide"
)

JOBS_FILE = "jobs_data.json"


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


# --- 2. 개인정보 자동 마스킹 ---
def mask_personal_info(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\b(\d{6})[- ]?([1-4]\d{6})\b", r"\1-*******", text)
    text = re.sub(
        r"\b(01[016789]|02|0[3-9]\d{1})[- ]?(\d{3,4})[- ]?(\d{4})\b",
        r"\1-****-\3",
        text,
    )
    text = re.sub(
        r"\b([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b",
        r"\1***@\2",
        text,
    )
    text = re.sub(
        r"(\([0-9]{5}\)\s*[가-힣]+(?:\s+[가-힣]+)?)\s+[가-힣0-9\s-]+",
        r"\1 [상세주소 마스킹]",
        text,
    )
    return text


# --- 3. 사람인 지원자 성명 100% 정밀 추출기 ---
def extract_candidate_name_robust(text: str, default_name: str) -> str:
    # 1순위: 사람인 문서 마지막 장의 법적 고지 항목 "작성자 : OOO"
    m_author = re.search(r"작성자\s*:\s*([가-힣]{2,4})\b", text)
    if m_author and m_author.group(1) not in [
        "사람인",
        "지원자",
        "이력서",
        "확인",
    ]:
        return m_author.group(1).strip()

    # 2순위: 별도 첨부파일 제출자의 파일명 링크 (예: 이솔지_이력서.pdf, 이제욱 이력서)
    m_file = re.search(r"([가-힣]{2,4})[_\s]이력서", text)
    if m_file and m_file.group(1) not in [
        "사람인",
        "지원자",
        "기본",
        "상세",
    ]:
        return m_file.group(1).strip()

    # 3순위: 입사지원일 바로 다음 줄에 붙어있는 성명 (예: 주아현경력, 정상진 경력)
    m_apply = re.search(
        r"입사지원일\s*:[^\n\r]+[\r\n\s]*([가-힣]{2,4})(?:경력|신입)", text
    )
    if m_apply:
        cand = m_apply.group(1).strip()
        if cand not in ["지원분야", "인적사항", "기본정보"]:
            return cand

    # 4순위: 성별, 연령 바로 윗줄 패턴 (지역명 필터링 포함)
    m_meta = re.search(
        r"([가-힣]{2,4})(?:경력|신입)?[\r\n\s]+(?:남|여),\s*(?:19|20)\d{2}", text
    )
    if m_meta:
        cand = m_meta.group(1).strip()
        exclude_words = [
            "서울",
            "경기",
            "인천",
            "전체",
            "지역",
            "희망",
            "서초",
            "강남",
            "인적",
            "기본",
        ]
        if not any(w in cand for w in exclude_words):
            return cand

    return default_name


# --- 4. 자기소개서 절단 및 핵심 이력(학력/경력/자격증) 추출기 ---
def extract_core_resume_content(text: str) -> str:
    """자기소개서 뒷부분을 제거하여 토큰을 대폭 절약하고 AI 응답 속도를 극대화합니다."""
    # 자기소개서가 시작되는 위치 탐색
    cut_markers = ["자기소개서", "자기소개\n", "1. 지원동기", "성장과정"]
    cut_pos = -1
    for m in cut_markers:
        pos = text.find(m)
        if pos != -1:
            if cut_pos == -1 or pos < cut_pos:
                cut_pos = pos

    # 자기소개서 이전까지만 슬라이싱 (학력, 경력기술서, 자격증, 나이 등 포함)
    core_text = text[:cut_pos] if cut_pos != -1 else text
    return core_text.strip()


# --- 5. 사람인 일괄 인쇄 PDF 정밀 분할기 ---
def split_candidates_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)
    if total_pages == 0:
        return []

    pages_text = [p.extract_text() or "" for p in reader.pages]
    split_indices = []

    for idx, txt in enumerate(pages_text):
        if idx == 0:
            split_indices.append(0)
            continue

        has_apply_date = bool(re.search(r"입사지원일\s*:", txt))
        has_apply_field = bool(re.search(r"지원분야\s*:", txt))
        has_profile_meta = bool(
            re.search(r"(?:남|여),\s*(?:19|20)\d{2}\s*\(\d{1,2}세\)", txt)
        )

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
                }
            )

    return candidate_chunks


# --- 6. 429 Quota 자동 복구 안정형 AI 호출 엔진 ---
def call_gemini_with_retry(prompt: str, max_retries: int = 5) -> str:
    models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    target_model = None
    for m in models:
        try:
            target_model = genai.GenerativeModel(m)
            break
        except Exception:
            continue

    if not target_model:
        target_model = genai.GenerativeModel("gemini-3.8-flash")

    last_error = ""
    for attempt in range(max_retries):
        try:
            res = target_model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            return res.text
        except Exception as e:
            last_error = str(e)
            err_lower = last_error.lower()

            # 429 Quota 에러 시 대기 후 재시도
            if any(
                k in err_lower
                for k in ["429", "quota", "resourceexhausted", "rate"]
            ):
                wait_sec = 6 * (attempt + 1)
                time.sleep(wait_sec)
                continue
            elif "404" in err_lower:
                target_model = genai.GenerativeModel("gemini-2.5-flash")
                time.sleep(2)
                continue
            else:
                time.sleep(3)

    raise Exception(f"AI 심사 지연: {last_error[:120]}")


# --- 사이드바 인증 ---
with st.sidebar:
    st.header("🔒 시스템 인증")
    env_password = st.secrets.get("TEAM_PASSWORD", "")
    env_api_key = st.secrets.get("GEMINI_API_KEY", "")

    if env_password:
        user_password = st.text_input(
            "팀 비밀번호", type="password", value=st.session_state.get("auth_pw", "")
        )
        if user_password != env_password:
            st.error("비밀번호 불일치")
            st.stop()
        st.session_state["auth_pw"] = user_password
        st.success("인증 완료")

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

# --- 메인 탭 ---
tab_screening, tab_job_manager = st.tabs(
    ["📄 이력서 대량 스크리닝", "⚙️ 공고 관리"]
)
jobs_dict = load_jobs()

# ==========================================
# 탭 1: 이력서 대량 스크리닝
# ==========================================
with tab_screening:
    st.title("📄 지원자 이력서 자동 스크리닝")

    if not jobs_dict:
        st.warning("먼저 '⚙️ 공고 관리' 탭에서 공고를 등록해 주세요.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_job_title = st.selectbox(
                "평가할 공고 선택", list(jobs_dict.keys())
            )
            job_info = jobs_dict[selected_job_title]
        with col2:
            uploaded_files = st.file_uploader(
                "사람인 통합 PDF 파일 업로드",
                type=["pdf"],
                accept_multiple_files=True,
            )

        start_btn = st.button("🚀 서류 스크리닝 시작", type="primary")

        if start_btn and uploaded_files:
            all_candidates = []
            with st.spinner("PDF 문서에서 개별 지원자 영역을 분할하는 중..."):
                for up_file in uploaded_files:
                    chunks = split_candidates_from_pdf(up_file)
                    all_candidates.extend(chunks)

            total_cands = len(all_candidates)
            if total_cands == 0:
                st.error("지원자를 인식할 수 없습니다.")
            else:
                st.success(
                    f"💡 총 **{total_cands}명**의 지원자를 감지했습니다. 심사를 시작합니다."
                )

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, cand in enumerate(all_candidates):
                    cand_text = cand["text"]
                    cand_name = extract_candidate_name_robust(
                        cand_text, f"지원자 {cand['candidate_seq']}번"
                    )

                    # [별도 이력서 첨부자 전용 판별]
                    # 단순 자격증 첨부가 아니라, '파일 미리 보기 영역은 인쇄 및 저장이 되지 않습니다'가 본문에 명시된 경우
                    is_resume_attachment = (
                        "파일 미리 보기" in cand_text
                        and "인쇄 및 저장이 되지 않습니다" in cand_text
                    ) or ("파일을 따로 다운로드하여 확인하시길 바랍니다" in cand_text)

                    if is_resume_attachment:
                        status_text.text(
                            f"[{idx+1}/{total_cands}] {cand_name} (별도 이력서 파일 첨부자 확인 완료)"
                        )
                        results.append(
                            {
                                "지원자명": cand_name,
                                "평가등급": "확인필요",
                                "필수요건_충족여부": "보완필요 (첨부확인)",
                                "우대요건_해당사항": "별도 이력서 제출",
                                "종합_평가사유": "사람인 기본 양식 대신 별도 첨부파일(이력서/자기소개서)을 등록한 지원자입니다. 원본 파일을 직접 열람해야 합니다.",
                                "메모_확인필요": "이력서 개별 확인 필요 (별도 첨부파일 제출자)",
                                "페이지": cand["page_range"],
                            }
                        )
                        progress_bar.progress((idx + 1) / total_cands)
                        continue

                    # 일반 지원자: 자기소개서 제외 후 핵심 이력만 압축 전달
                    status_text.text(
                        f"[{idx+1}/{total_cands}] {cand_name} 서류 심사 중..."
                    )

                    core_text = extract_core_resume_content(cand_text)
                    masked_text = mask_personal_info(core_text)

                    eval_prompt = f"""
당신은 기업의 전문 채용 심사관입니다. 아래 공고 기준과 지원자의 핵심 이력(학력/학점/경력/자격증)을 대조 평가하여 순수한 JSON 객체로만 답하세요. 사족은 일체 생략하세요.

[공고 기준]
- 공고명: {selected_job_title}
- 담당 업무: {job_info.get('duties')}
- 필수 요건: {job_info.get('required')}
- 우대 요건: {job_info.get('preferred')}

[평가 기준]
- S: 필수 완벽 충족 + 우대 다수 충족 + 직무 역량 뛰어남
- A: 필수 충족 + 우대 일부 충족
- B: 필수 충족 + 우대 미미
- C: 필수 충족 애매하거나 직무 연관성 낮음
- 탈락: 필수 요건 명백히 미충족

[학점 규칙]
학점 3.0 이상 권장. 단, 학점 미기재 시 절대 탈락시키지 말고 메모_확인필요에 '이력서 확인 필요(학점 미기재)'라고 적고 다른 자격/경력을 중심으로 평가하세요.

[지원자 핵심 이력 요약]
{masked_text[:5000]}

[출력 JSON]
{{
  "지원자명": "{cand_name}",
  "평가등급": "S, A, B, C, 탈락 중 택1",
  "필수요건_충족여부": "충족 / 미충족 / 보완필요 중 택1",
  "우대요건_해당사항": "충족 우대요건 요약",
  "종합_평가사유": "등급 산정 핵심 사유 2문장",
  "메모_확인필요": "학점 미기재 여부 등 특이점 (없으면 '해당 없음')"
}}
"""
                    try:
                        raw_json = call_gemini_with_retry(eval_prompt)
                        parsed = json.loads(raw_json)
                        parsed["지원자명"] = cand_name
                        parsed["페이지"] = cand["page_range"]
                        results.append(parsed)
                    except Exception as e:
                        results.append(
                            {
                                "지원자명": cand_name,
                                "평가등급": "재확인",
                                "필수요건_충족여부": "미확인",
                                "우대요건_해당사항": "-",
                                "종합_평가사유": f"분석 오류: {str(e)}",
                                "메모_확인필요": "수동 확인 요망",
                                "페이지": cand["page_range"],
                            }
                        )

                    progress_bar.progress((idx + 1) / total_cands)
                    # 무료 API 분당 할당량(RPM) 보호를 위한 4초 간격 유지
                    time.sleep(4.0)

                status_text.text(
                    f"✅ 총 {total_cands}명의 스크리닝이 모두 완료되었습니다!"
                )

                # 파일명 컬럼을 완전히 제외한 데이터프레임 구성
                df = pd.DataFrame(results)
                cols_order = [
                    "지원자명",
                    "평가등급",
                    "필수요건_충족여부",
                    "우대요건_해당사항",
                    "종합_평가사유",
                    "메모_확인필요",
                    "페이지",
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
# 탭 2: 공고 관리
# ==========================================
with tab_job_manager:
    st.title("⚙️ 채용 공고 관리")
    st.subheader("➕ 새 공고 등록")
    raw_job_text = st.text_area(
        "사람인 공고 본문 붙여넣기",
        height=180,
        placeholder="채용 공고 본문을 복사하여 여기에 붙여넣으세요...",
    )

    if st.button("🤖 AI로 공고 기준 자동 추출 및 등록"):
        if not raw_job_text.strip():
            st.error("공고 텍스트를 입력해 주세요.")
        else:
            with st.spinner("AI가 공고 요건을 파악하고 있습니다..."):
                extract_prompt = f"""
채용 사이트 본문에서 공고명, 주요 업무, 지원 필수 자격, 우대 사항을 추출해 JSON으로 답하세요.
{raw_job_text[:6000]}

[출력 형식]
{{
  "title": "공고명",
  "duties": "담당 업무 요약",
  "required": "필수 자격 요약",
  "preferred": "우대 사항 요약"
}}
"""
                try:
                    res_json = call_gemini_with_retry(extract_prompt)
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
                    key=f"d_{title}",
                    disabled=True,
                )
                st.text_area(
                    "필수 요건",
                    value=info.get("required", ""),
                    key=f"r_{title}",
                    disabled=True,
                )
                st.text_area(
                    "우대 사항",
                    value=info.get("preferred", ""),
                    key=f"p_{title}",
                    disabled=True,
                )
                if st.button("🗑️ 이 공고 삭제", key=f"del_{title}"):
                    del jobs_dict[title]
                    save_jobs(jobs_dict)
                    st.warning(f"'{title}' 공고가 삭제되었습니다.")
                    st.rerun()
    else:
        st.info("현재 등록된 공고가 없습니다.")
