import json
import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.constants import START
from langgraph.graph import StateGraph, END

# ---------- ENV ----------
load_dotenv()
if not os.getenv("MISTRAL_API_KEY"):
    raise RuntimeError("Нет MISTRAL_API_KEY в .env")

# ---------- TWO MODELS (can be same model, different temps) ----------
llm_tech = ChatMistralAI(model="mistral-large-latest", temperature=0.2)
llm_interviewer = ChatMistralAI(model="mistral-large-latest", temperature=0.5)

TECH_SYS = """Ты агент Tech. Ты помогаешь интервьюеру вести техсобеседование.

Вход:
- Инфо о кандидате (role/grade/exp могут быть пустыми)
- План тем (строка). Если пуст — создай план из 6–10 тем под role/grade.
- Последний вопрос интервьюера
- Последний ответ кандидата
- (опционально) несколько последних сообщений диалога

Твои задачи:
1) Если plan пуст: создай план в формате "No.1 <topic> - <level>" ... "No.N ..."
2) Проанализируй последний ответ кандидата:
   - если ответ слабый/уходит от сути — задай уточняющий вопрос по этому же месту
   - если ответ сильный — переходи к следующей теме плана
3) Адаптируй сложность:
   - сильный ответ → вопрос глубже (trade-offs, ограничения, edge cases)
   - слабый ответ → проще/с подсказкой, но всё равно один вопрос

Проверки поведения кандидата:
- Если кандидат задаёт встречный вопрос интервьюеру (про компанию/задачи/стек) — верни intent="candidate_question" и подготовь короткий ответ для интервьюера.
- Если кандидат уходит в оффтоп — intent="offtopic".
- Если кандидат уверенно утверждает сомнительную “фактическую” вещь (вымышленная версия, несуществующий термин) — intent="hallucination".

Формат ответа: СТРОГО JSON без markdown:
{
  "plan": "<строка>",
  "intent": "normal|candidate_question|offtopic|hallucination",
  "answer_to_candidate": "<строка или пусто>",
  "query": "<РОВНО ОДИН вопрос, одна строка, заканчивается ?>",
  "done": false,
  "recommendation": "<кратко: strong|ok|weak>"
}
Если done=true: query="Техническое интервью окончено."
"""


INTERVIEWER_SYS = """Ты интервьюер. Сформируй следующее сообщение кандидату.

Тебе даны:
- Совет техника (может включать intent и answer_to_candidate)
- Совет контролёра (какое поле анкеты добрать)
- Вся история диалога

Правила:
- 1–3 предложения, дружелюбно, без канцелярита.
- Если control_advice просит заполнить анкету — спроси это.
- Если tech intent == "candidate_question": сначала ответь коротко на вопрос кандидата (1–2 предложения), затем задай следующий вопрос по интервью.
- Если tech intent == "offtopic": мягко верни к интервью и задай вопрос.
- Если tech intent == "hallucination": вежливо отметь, что утверждение спорное/неверное и попроси объяснить основы/привести доказательства, затем продолжай по теме.
- Никаких префиксов, только текст кандидату.
"""
OBS_SYS = """Ты агент Observer. Ты пишешь финальный отчет по интервью в формате Markdown.

Вход:
- Полный диалог интервьюера и кандидата
- Карточка кандидата (name/role/grade/exp) и план тем
- Технические заметки (tech_advice, recommendation) — могут быть неполными

Критические правила:
- НЕ выдумывай. Если данных нет — пиши "не обсуждалось" / "нет данных в диалоге".
- Ссылайся на конкретные фрагменты диалога (коротко, 5–15 слов, без длинных цитат).
- Пиши только Markdown, НЕ оборачивай в ```.

Структура отчета (строго следуй):

# Final feedback

## Candidate
- Name: ...
- Target role: ...
- Target grade: ...
- Experience: ...

## Verdict
- Recommendation: Hire / Hold / No Hire
- Confidence: 0-100
- Reasons:
  - ...
  - ...
  - ...

## Hard skills
### Strengths
- ...
### Gaps & correct answer
- Topic: ...
  - What was said: "..."
  - Why it's a gap: ...
  - How it should be: ...

## Soft skills
- Communication: ...
- Structure: ...
- Handling feedback/questions: ...

## Risks / Unknowns
- ...

## 4-week roadmap
Week 1:
- ...
Week 2:
- ...
Week 3:
- ...
Week 4:
- ...

## Notes
- Any extra concise notes.
"""
CONTROL_SYS = """Ты агент Control. Ты обновляешь карточку кандидата на основе диалога.

Вход:
- История переписки интервьюер ↔ кандидат
- Текущая карточка кандидата (ей доверяй)

Правила:
- НЕ выдумывай факты. Заполняй поля только если кандидат явно сообщил это.
- Уже заполненные поля НЕ меняй, кроме случая когда кандидат явно исправил.
- role и grade нормализуй к общеупотребимым вариантам (например: "Backend Python", "Middle").
- exp: если кандидат сказал опыт — кратко перефразируй; если не сказал — сформируй короткий вопрос для уточнения.

Верни СТРОГО валидный JSON без markdown:
{
  "info": {"name": "", "role": "", "grade": "", "exp": ""},
  "done": 0.0,
  "query": ""
}

Где:
- done — доля заполненности обязательных полей (name, role, grade). exp не обязателен.
- query — ОДНО короткое предложение, что спросить дальше, чтобы заполнить пустое.
Если обязательные поля заполнены, query = "Можно переходить к техническим вопросам."

"""

STOP = False
# ---------- STATE ----------
class S(TypedDict):
    history: List[BaseMessage]
    tech_advice: str
    interviewer_msg: str
    control_advice: str
    candidate_info: dict
    done_info: float
    done_tech: float
    plan: str
    tech_recom: str
    observer: str

    participant_name: str
    scenario_id: int
    turn_id: int
    turns: list
    current_question: str
    final_feedback: str


import re

def extract_json(text: str) -> dict:
    if text is None:
        return ""

    t = text.strip()

    # Если текст начинается с ``` (возможно ```json / ```JSON и т.п.)
    if t.startswith("```"):
        # убираем первую строку целиком: ``` или ```json
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", t)
        # убираем закрывающие ```
        t = re.sub(r"\n?```$", "", t)

    return json.loads(t.strip())

def build_internal_thoughts(state: S) -> str:
    # Формат строго: [agent]: ...\n
    lines = []
    if check_done_info(state) == 1:
        lines.append(f"[Control]: {state.get('control_advice','')}\n")
    elif check_done_info(state) == 2:
        lines.append(f"[Tech]: {state.get('tech_advice','')}\n")
    # observer иногда пустой до конца — это ок
    else:
        lines.append(f"[Observer]: {state.get('observer','')}\n")
    return "".join(lines)

def append_turn(state: S, asked_question: str, user_message: str, internal_thoughts: str):
    state["turn_id"] += 1
    state["turns"].append({
        "turn_id": state["turn_id"],
        "agent_visible_message": asked_question,
        "user_message": user_message,
        "internal_thoughts": internal_thoughts
    })

def save_json_log(state: S):
    filename = f"interview_log_{state['scenario_id']}.json"
    payload = {
        "participant_name": state["participant_name"],
        "turns": state["turns"],
        "final_feedback": state.get("final_feedback","")
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return filename

def save_md_feedback(state: S):
    filename = f"final_feedback_{state['scenario_id']}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(state.get("final_feedback","").strip() + "\n")
    return filename


def control_agent(state: S) -> S:
    info = state['candidate_info']
    resp = llm_tech.invoke([
        SystemMessage(content=CONTROL_SYS),
        HumanMessage(content=f"Диалог: {state['history']} \n\nТекущая информация:\n{info}")
    ])
    try:
        state['candidate_info'] = extract_json(resp.content)['info']
        state['control_advice'] = extract_json(resp.content)['query']
        state['done_info'] = float(extract_json(resp.content)['done'])
    except Exception as e:
        print("РУХНУЛ JSON в контроле")
        print(resp.content)

        state['control_advice'] = ""
    return state

# ---------- NODES ----------
def tech_node(state: S) -> S:
    # берем последний ответ пользователя из истории
    last_user = ""
    for m in reversed(state["history"]):
        if isinstance(m, HumanMessage):
            last_user = m.content
            break
    last_ai = ""
    for m in reversed(state["history"]):
        if isinstance(m, AIMessage):
            last_ai = m.content
            break

    resp = llm_tech.invoke([
        SystemMessage(content=TECH_SYS),
        HumanMessage(content=f"План:\n{state['plan']}"
                             f"Последнее сообщение интервьюера:\n{last_ai}\n\n"
                             f"Последний ответ кандидата:\n{last_user}")
    ])
    state["tech_advice"] = extract_json(resp.content)['query']
    state["done_tech"] = extract_json(resp.content)['done']
    state["plan"] = extract_json(resp.content)['plan']
    state["tech_recom"] = extract_json(resp.content)['recommendation']
    return state

def obs_agent(state: S) -> S:

    resp = llm_tech.invoke([
        SystemMessage(content=OBS_SYS),
        HumanMessage(content=
                     f"Карточка кандидата:\n{state['candidate_info']}\n\n"
                     f"План техтем:\n{state['plan']}\n\n"
                     f"Тех-рекомендация:\n{state['tech_recom']}\n\n"
                     f"Диалог:\n{state['history']}\n\n"
                     f"Сформируй финальный отчет."
                     )

    ])
    state['observer'] = resp.content
    return state

def check_done_info(state: S) -> float:
    if state["done_info"]>=0.9:
        if state["done_tech"]==1:
            return 3
        else:
            return 2
    else:
        return 1
def check_done_tech(state: S) -> float:
    return bool(state["done_tech"]==1)

def interviewer_node(state: S) -> S:
    prefix = ""
    stage = check_done_info(state)
    if stage == 1:
        prefix += f"Совет контролёра:\n{state['control_advice']}\n\n"
    elif stage == 2:
        prefix += f"Совет техника:\n{state['tech_advice']}\n\n"
    resp = llm_interviewer.invoke([
        SystemMessage(content=INTERVIEWER_SYS),
        *state["history"],
        HumanMessage(content=prefix+"На основе советов сформируй следующее сообщение.")
    ])
    msg = (resp.content or "").strip()
    state["interviewer_msg"] = msg
    state["history"].append(AIMessage(content=msg))
    return state


# ---------- GRAPH ----------
def build_graph():
    g = StateGraph(S)
    g.add_node("tech", tech_node)
    g.add_node("interviewer", interviewer_node)
    g.add_node("control", control_agent)
    g.add_node("observer", obs_agent)

    g.add_conditional_edges(
        START,
        check_done_info,
        {
            3: "observer",
            2: "tech",
            1: "control"
        }
    )
    g.add_edge( "tech", "interviewer")
    g.add_edge( "control", "interviewer")

    g.add_edge("interviewer", END)
    g.add_edge("observer", END)
    return g.compile()


def main():
    graph = build_graph()

    scenario_id = int(input("scenario_id (1..5): ").strip() or "1")
    participant_name = input("Ваше ФИО (participant_name): ").strip()

    state: S = {
        "history": [],
        "tech_advice": "",
        "control_advice": "",
        "interviewer_msg": "",
        "candidate_info": {},
        "done_info": 0,
        "done_tech": 0,
        "plan": "",
        "tech_recom": "",
        "observer": "",

        # --- ДОБАВИЛИ ---
        "scenario_id": scenario_id,
        "participant_name": participant_name,
        "turn_id": 0,
        "turns": [],
        "current_question": "",
        "final_feedback": ""
    }

    # фиксированное первое сообщение
    first = "Привет, расскажи о себе"
    print(f"[Interviewer]: {first}")
    state["history"].append(AIMessage(content=first))
    state["current_question"] = first  # важно для логгера

    while True:
        user = input("[You]: ").strip()
        if not user:
            continue

        asked_question = state["current_question"]

        # добавляем ответ кандидата в историю
        state["history"].append(HumanMessage(content=user))

        is_stop = user.lower() in {"стоп", "stop"}

        if is_stop:
            # заставляем систему перейти в observer
            state["done_info"] = 1
            state["done_tech"] = 1

            state = graph.invoke(state)  # observer заполнит state["observer"]

            # final_feedback берём из observer
            state["final_feedback"] = state.get("observer", "").strip()

            internal_thoughts = build_internal_thoughts(state)
            append_turn(state, asked_question, user, internal_thoughts)

            json_file = save_json_log(state)
            md_file = save_md_feedback(state)

            print("[Interviewer]: Спасибо! Я сохранил отчет.")
            print(f"JSON: {json_file}")
            print(f"MD: {md_file}")
            break

        # обычный ход
        state = graph.invoke(state)
        if check_done_info(state)==2:
            internal_thoughts = build_internal_thoughts(state)
            append_turn(state, asked_question, user, internal_thoughts)

        # следующий вопрос интервьюера
        print(f"[Interviewer]: {state['interviewer_msg']}")
        state["current_question"] = state["interviewer_msg"]


if __name__ == "__main__":
    main()
