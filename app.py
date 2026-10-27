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


# --- 개인정보 자동 마스킹 (보안 필터) ---
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


# --- 사람인 지원자 이름 고속 추출기 ---
def extract_candidate_name(text: str, default_name: str) -> str:
    # 1. "정상진 경력", "이솔지 신입" 패턴 매칭
    m = re.search(r"([가-힣]{2,4})\s+(?:경력|신입)", text)
    if m:
        return m.group(1)
    # 2. 파일명 링크 패턴 매칭 (이솔지_이력서.pdf 등)
    m = re.search(r"([가-힣]{2,4})[_\s]이력서", text)
    if m:
        return m.group(1)
    # 3. 성별/나이 앞 성명 매칭
    m = re.search(r"([가-힣]{2,4})\s*[\n\r]+\s*(?:남|여),", text)
    if m:
        return m.group(1)
    return default_name


# --- 사람인 일괄 인쇄 PDF 정밀 분할기 ---
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
                    "filename": uploaded_file.name,
                }
            )

    return candidate_chunks


# --- 초고속 Gemini 호출 엔진 ---
def call_gemini_fast(prompt: str) -> str:
    models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            # 불필요한 추론 토큰 지연 방지
            res = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            return res.text
        except Exception as e:
            if "404" in str(e).lower():
                continue
            time.sleep(1.5)
            # 1회 재시도
            try:
                res = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1,
                    },
                )
                return res.text
            except Exception:
                continue
    raise Exception("AI 응답 지연 (잠시 후 다시 시도해 주세요)")


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
    st.subheader("🔑 AI 키")
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

# --- 메인 탭 구성 ---
tab_screening, tab_job_manager = st.tabs(
    ["📄 이력서 대량 스크리닝", "⚙️ 공고 관리"]
)
jobs_dict = load_jobs()

# ==========================================
# 탭 1: 이력서 대량 스크리닝
# ==========================================
with tab_screening:
    st.title("📄 지원자 이력서 초고속 스크리닝")

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
            with st.spinner("PDF 문서에서 지원자 영역 분할 중..."):
                for up_file in uploaded_files:
                    chunks = split_candidates_from_pdf(up_file)
                    all_candidates.extend(chunks)

            total_cands = len(all_candidates)
            if total_cands == 0:
                st.error("이력서 텍스트를 읽을 수 없습니다.")
            else:
                st.success(f"💡 총 {total_cands}명의 지원자를 감지했습니다.")

                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, cand in enumerate(all_candidates):
                    cand_text = cand["text"]
                    cand_name = extract_candidate_name(
                        cand_text, f"지원자 {cand['candidate_seq']}번"
                    )

                    # [핵심 로직] 사람인 별도 첨부파일 제출자 감지 (AI 호출 생략 -> 0.01초 처리)
                    is_attachment_only = (
                        "파일 미리 보기" in cand_text
                        and "인쇄 및 저장이 되지 않습니다" in cand_text
                    ) or ("파일을 따로 다운로드하여 확인하시길 바랍니다" in cand_text)

                    if is_attachment_only:
                        status_text.text(
                            f"[{idx+1}/{total_cands}] {cand_name} (별도 첨부파일 확인 필요 - 고속 처리 완료)"
                        )
                        results.append(
                            {
                                "지원자명": cand_name,
                                "평가등급": "확인필요",
                                "필수요건_충족여부": "보완필요 (첨부확인)",
                                "우대요건_해당사항": "별도 서류 제출",
                                "종합_평가사유": "사람인 기본 웹 양식 대신 별도 첨부파일(이력서/자기소개서)을 등록한 지원자입니다. 원본 파일을 직접 확인해야 합니다.",
                                "메모_확인필요": "이력서 개별 확인 필요 (별도 첨부파일 제출자)",
                                "페이지": cand["page_range"],
                                "파일명": cand["filename"],
                            }
                        )
                        progress_bar.progress((idx + 1) / total_cands)
                        continue

                    # 일반 지원자: 핵심 텍스트만 추출 후 AI 분석
                    status_text.text(
                        f"[{idx+1}/{total_cands}] {cand_name} 분석 중..."
                    )
                    masked_text = mask_personal_info(cand_text)

                    eval_prompt = f"""
당신은 채용 심사관입니다. 지원자 이력서를 분석해 JSON으로만 답하세요. 긴 생각이나 사족은 생략하고 핵심만 신속히 작성하세요.

[공고 기준]
- 공고명: {selected_job_title}
- 담당 업무: {job_info.get('duties')}
- 필수 요건: {job_info.get('required')}
- 우대 요건: {job_info.get('preferred')}

[평가 등급]
- S: 필수 완벽 충족 + 우대 다수 충족
- A: 필수 충족 + 우대 일부 충족
- B: 필수 충족 + 우대 미미
- C: 필수 충족 애매하거나 직무 핏 낮음
- 탈락: 필수 요건 미충족

[학점 규칙]
학점 3.0 이상 권장. 단, 학점 미기재 시 절대 탈락시키지 말고 메모_확인필요에 '이력서 확인 필요(학점 미기재)' 명시.

[지원자 이력서]
{masked_text[:5000]}

[출력 JSON]
{{
  "지원자명": "{cand_name}",
  "평가등급": "S, A, B, C, 탈락 중 1개",
  "필수요건_충족여부": "충족 / 미충족 / 보완필요 중 1개",
  "우대요건_해당사항": "충족 우대요건 요약",
  "종합_평가사유": "핵심 사유 1~2문장",
  "메모_확인필요": "학점 미기재 여부 등 특이점 (없으면 '해당 없음')"
}}
"""
                    try:
                        raw_json = call_gemini_fast(eval_prompt)
                        parsed = json.loads(raw_json)
                        parsed["지원자명"] = cand_name
                        parsed["페이지"] = cand["page_range"]
                        parsed["파일명"] = cand["filename"]
                        results.append(parsed)
                    except Exception as e:
                        results.append(
                            {
                                "지원자명": cand_name,
                                "평가등급": "재확인",
                                "필수요건_충족여부": "미확인",
                                "우대요건_해당사항": "-",
                                "종합_평가사유": f"분석 일시 오류: {str(e)}",
                                "메모_확인필요": "수동 확인 요망",
                                "페이지": cand["page_range"],
                                "파일명": cand["filename"],
                            }
                        )

                    progress_bar.progress((idx + 1) / total_cands)
                    time.sleep(0.5)

                status_text.text(
                    f"✅ 총 {total_cands}명의 스크리닝이 완료되었습니다!"
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
            with st.spinner("AI가 공고 요건을 추출하고 있습니다..."):
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
                    res_json = call_gemini_fast(extract_prompt)
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
