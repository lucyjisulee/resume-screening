import io
import json
import os
import re
import google.generativeai as genai
import pandas as pd
from pypdf import PdfReader
import streamlit as st

# --- 기본 페이지 설정 ---
st.set_page_config(
    page_title="AI 채용 서류 스크리닝 시스템", page_icon="📋", layout="wide"
)

# 공고 데이터 저장 파일 경로
JOBS_FILE = "jobs_data.json"


# --- 공고 저장/불러오기 유틸 ---
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


# --- 개인정보 자동 마스킹 함수 (보안 필터) ---
def mask_personal_info(text: str) -> str:
    # 1. 주민등록번호 패턴 마스킹 (예: 950101-1234567 -> 950101-*******)
    text = re.sub(r"(\d{6})[- ]?([1-4]\d{6})", r"\1-*******", text)

    # 2. 전화번호 및 휴대전화 마스킹 (예: 010-1234-5678 -> 010-****-5678)
    text = re.sub(
        r"(01[016789]|02|0[3-9]\d{1})[- ]?(\d{3,4})[- ]?(\d{4})",
        r"\1-****-\3",
        text,
    )

    # 3. 이메일 주소 마스킹 (예: recruit@company.com -> r***@company.com)
    text = re.sub(
        r"([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
        r"\1***@\2",
        text,
    )

    return text


# --- PDF 텍스트 추출 함수 ---
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text.strip()


# --- 사이드바: 팀 비밀번호 및 API 키 관리 ---
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
        st.info("💡 배포 시 Secrets에 TEAM_PASSWORD를 설정하면 암호가 활성화됩니다.")

    st.markdown("---")
    st.subheader("🔑 AI 엔진 설정")
    api_key = env_api_key or st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.get("api_key", ""),
    )

    if api_key:
        st.session_state["api_key"] = api_key
        genai.configure(api_key=api_key)
    else:
        st.warning("Gemini API 키를 입력하거나 Secrets에 등록해 주세요.")
        st.stop()

# --- 메인 탭 구성 ---
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
        "개인정보(주민번호, 전화번호, 이메일)는 AI 분석 전 자동으로 비식별 마스킹 처리됩니다."
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
            uploaded_resumes = st.file_uploader(
                "지원자 이력서 PDF 다중 업로드 (드래그 앤 드롭 가능)",
                type=["pdf"],
                accept_multiple_files=True,
            )

        start_btn = st.button("🚀 서류 스크리닝 시작", type="primary")

        if start_btn and uploaded_resumes:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            model = genai.GenerativeModel("gemini-3.8-flash")

            for idx, resume_file in enumerate(uploaded_resumes):
                status_text.text(
                    f"[{idx+1}/{len(uploaded_resumes)}] {resume_file.name} 분석 중 (보안 마스킹 적용)..."
                )

                try:
                    # PDF 텍스트 추출 후 즉시 개인 식별 정보 마스킹 적용
                    raw_text = extract_pdf_text(resume_file)
                    masked_resume_text = mask_personal_info(raw_text)

                    prompt = f"""
당신은 기업의 전문 인사담당자(채용 심사관)입니다.
아래 채용 공고 기준과 지원자 이력서를 비교 분석하여 평가 결과를 반드시 지정된 JSON 형식으로만 응답하세요.

[채용 공고 기준]
- 공고명: {selected_job_title}
- 담당 업무: {job_info.get('duties')}
- 필수 요건: {job_info.get('required')}
- 우대 요건: {job_info.get('preferred')}

[평가 및 등급 부여 규칙]
1. 등급 체계:
   - S: 필수 요건 완벽 충족 + 우대 사항 대부분 충족 + 직무 역량 뛰어남
   - A: 필수 요건 충족 + 우대 사항 일부 충족
   - B: 필수 요건 기본 충족 + 우대 사항 미미
   - C: 필수 요건 충족이 다소 애매하거나 직무 핏이 낮음
   - 탈락: 필수 요건을 명백히 충족하지 못함
2. 학점(GPA) 특별 규칙:
   - 기본적으로 학점 3.0 이상을 권장하나, 이력서에 학점이 아예 기재되어 있지 않은 경우 절대로 탈락 사유로 삼지 마세요.
   - 학점이 기재되어 있지 않다면 '메모_확인필요' 항목에 반드시 '이력서 확인 필요(학점 미기재)'라고 명시하고, 다른 필수 요건을 중심으로 등급을 산정하세요.

[지원자 이력서 원문 (개인 식별정보 마스킹됨)]
{masked_resume_text[:12000]}

[출력 형식]
반드시 유효한 JSON 객체만 반환하세요.
{{
  "지원자명": "이름 (마스킹으로 추정 불가 시 '미상')",
  "평가등급": "S, A, B, C, 탈락 중 택 1",
  "필수요건_충족여부": "충족 / 미충족 / 보완필요 중 택 1",
  "우대요건_해당사항": "충족하는 우대 요건 요약",
  "종합_평가사유": "서류 합격/불합격 및 등급 부여 핵심 사유 (2~3문장)",
  "메모_확인필요": "학점 미기재 확인 필요 여부 또는 서류상 확인이 필요한 특이점 (없으면 '해당 없음')"
}}
"""
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "response_mime_type": "application/json"
                        },
                    )
                    parsed_res = json.loads(response.text)
                    parsed_res["파일명"] = resume_file.name
                    results.append(parsed_res)

                except Exception as e:
                    results.append(
                        {
                            "파일명": resume_file.name,
                            "지원자명": "오류",
                            "평가등급": "분석 실패",
                            "필수요건_충족여부": "오류",
                            "우대요건_해당사항": "-",
                            "종합_평가사유": f"분석 에러: {str(e)}",
                            "메모_확인필요": "재확인 필요",
                        }
                    )

                progress_bar.progress((idx + 1) / len(uploaded_resumes))

            status_text.text("✅ 모든 지원자 서류 스크리닝이 완료되었습니다!")

            df = pd.DataFrame(results)
            column_order = [
                "지원자명",
                "평가등급",
                "필수요건_충족여부",
                "우대요건_해당사항",
                "종합_평가사유",
                "메모_확인필요",
                "파일명",
            ]
            df = df[[col for col in column_order if col in df.columns]]

            st.subheader("📊 스크리닝 결과")
            st.dataframe(df, use_container_width=True)

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="스크리닝결과")
            excel_data = excel_buffer.getvalue()

            st.download_button(
                label="📥 평가 결과 엑셀(.xlsx) 다운로드",
                data=excel_data,
                file_name=f"{selected_job_title}_서류스크리닝결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ==========================================
# 탭 2: 공고 관리 (매주 유동적 추가/삭제/AI 추출)
# ==========================================
with tab_job_manager:
    st.title("⚙️ 채용 공고 관리")
    st.caption(
        "사람인 공고 본문을 복사해 붙여넣으면 AI가 기준을 추출해 목록에 추가합니다."
    )

    st.subheader("➕ 새 공고 등록 (텍스트 복사-붙여넣기)")
    raw_job_text = st.text_area(
        "사람인 공고 본문 전체 붙여넣기 (Ctrl+A -> Ctrl+C -> 여기에 Ctrl+V)",
        height=180,
        placeholder="채용 공고 웹페이지 본문 글자를 전체 복사해 여기에 붙여넣으세요...",
    )

    if st.button("🤖 AI로 공고 기준 자동 추출 및 등록"):
        if not raw_job_text.strip():
            st.error("공고 텍스트를 입력해 주세요.")
        else:
            with st.spinner("AI가 공고 요건을 파악하고 있습니다..."):
                try:
                    extractor_model = genai.GenerativeModel("gemini-3.8-flash")
                    extract_prompt = f"""
채용 사이트에서 복사한 원문 텍스트입니다.
공고명, 주요 업무, 지원 필수 자격, 우대 사항을 추출하여 순수한 JSON 형식으로 반환하세요.

[원문 텍스트]
{raw_job_text[:10000]}

[출력 형식]
{{
  "title": "공고명 (예: 백엔드 개발자 3년 이상)",
  "duties": "담당 업무 항목 요약",
  "required": "필수 자격 요건 요약",
  "preferred": "우대 사항 요약"
}}
"""
                    res = extractor_model.generate_content(
                        extract_prompt,
                        generation_config={
                            "response_mime_type": "application/json"
                        },
                    )
                    parsed_job = json.loads(res.text)

                    job_title = parsed_job.get("title", "새 채용공고")
                    jobs_dict[job_title] = {
                        "duties": parsed_job.get("duties", ""),
                        "required": parsed_job.get("required", ""),
                        "preferred": parsed_job.get("preferred", ""),
                    }
                    save_jobs(jobs_dict)
                    st.success(f"'{job_title}' 공고가 성공적으로 등록되었습니다!")
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
