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

TECH_SYS = """Ты тех-помощник интервьюера. Твоя цель — помочь задать следующий технический вопрос.

Тебе будут даны:
1) Информация о кандидате (может быть неполной)
2) План технических вопросов по темам в формате: "No.X <Тема> - <уровень>"
3) Последний вопрос интервьюера
4) Последний ответ кандидата

Задачи:
- Если plan пустой или отсутствует: составь plan из 6–10 тем, релевантных role/grade кандидата (если role/grade неизвестны — составь универсальный план).
- Если plan уже есть: НЕ переписывай его полностью. Можно только:
  (a) отметить текущий номер, на котором вы находитесь (логически),
  (b) добавить 1–2 темы в конец, если явно не хватает.
- Проанализируй последний ответ кандидата и выбери следующий шаг по плану.

Правила для вопроса:
- query должен содержать РОВНО ОДИН вопрос (одна строка, один знак вопроса в конце).
- Вопрос должен либо закрывать слабое место ответа, либо идти по следующей теме плана.
- Проси конкретику: пример, метрики, компромиссы, причины, результат, ограничения.
- Не задавай два вопроса сразу. Не используй списки, markdown, нумерацию в query.
- Можно давить на слабые места и быстрее проходить сильные.

Условие завершения:
- done = true только если по плану уже пройдены все темы (или интервьюер явно завершил тех. часть).
- Если done = true: query = "Техническое интервью окончено."
- Технический итог, соответствие поз ции в "recommendation"

Формат ответа: СТРОГО валидный JSON, без markdown, без комментариев, без лишних ключей.
{
  "plan": "<строка>",
  "query": "<строка>",
  "done": false,
  "recommendation": str
}
"""


INTERVIEWER_SYS = """Ты интервьюер. Твоя задача — написать следующее сообщение кандидату.
Тебе будут даны:
1) Совет техника (о качестве ответа и следующем вопросе)
2) Совет коллеги (какое поле анкеты надо добрать)
3) Весь диалог с кандадатом

Правила:
- Пиши дружелюбно и коротко: 1–3 предложения.
- Если совет коллеги просит добрать поле анкеты — приоритезируй это.

Выводи только текст сообщения кандидату, без префиксов.
"""
OBS_SYS = """Твоя задача по входным данным решить нанимать кандидата или нет"""
CONTROL_SYS = """Ты контролируешь интервью и заполняешь карточку кандидата.
У тебя НЕТ прямого контакта с кандидатом: ты работаешь только по текстам.

Тебе дано:
- История переписки интервьюера и кандидата
- Текущая информация (ей доверяй). НЕ МЕНЯЙ заполненные поля, если кандидат явно не исправил их.

Задача:
1) Извлеки/уточни поля name, role, grade, exp из ответа кандидата.
2) Заполни info:
   - name, role, grade — обязательны. Если кандидат не сказал, оставь "" (пустая строка).
   - exp — уточни про опыт хотя бы 1 раз, попытайся узнать, но не настаивай. Если после попытки кандидат не хочет рассказывать, заполни поле своими словами.
3) Если ты только что спросил про опыт, пропусти. Посчитай done = (процент заполненности карточки от 0 до 1). 
4) query: короткая подсказка интервьюеру, что спросить, чтобы заполнить ПУСТЫЕ обязательные поля.
   - Если done==1, query = "Можно переходить к техническим вопросам."

Критически важно:
- НИЧЕГО не выдумывай. Только то, что явно есть в тексте кандидата. Преобразуй role, grade к общемировым нормам
- Возвращай СТРОГО валидный JSON. Без markdown. Без комментариев. Без лишних ключей.

Формат (строго):
{
  "info": {
    "name": "",
    "role": "",
    "grade": "",
    "exp": ""
  },
  "done": 0.0,
  "query": ""
}
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
    print(resp.content)
    state["tech_advice"] = extract_json(resp.content)['query']
    state["done_tech"] = extract_json(resp.content)['done']
    state["plan"] = extract_json(resp.content)['plan']
    state["tech_recom"] = extract_json(resp.content)['recommendation']
    return state

def obs_agent(state: S) -> S:

    resp = llm_tech.invoke([
        SystemMessage(content=OBS_SYS),
        HumanMessage(content=
                     f"Диалог:\n{state['history']}\n\n"
                     f"Рекомендация техника: \n{state['tech_advice']}\n\nСоставь подробны отчёт")
    ])
    print("ОБСЕРВЕР++++++++++++++++++++++++++++++++++++++++++++++++")
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
    if not check_done_info(state):
        prefix +=f"Совет коллеги (отвечает за заполнение формы о кандидате):\n{state['control_advice']}\n\n"
    elif not check_done_tech(state):
        prefix += f"Совет техника:\n{state['tech_advice']}\n\n"
    print(prefix)
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
        "observer": ""
    }

    # фиксированное первое сообщение
    first = "Привет, расскажи о себе"
    print(f"[Interviewer]: {first}")
    state["history"].append(AIMessage(content=first))

    while True:
        user = input("[You]: ").strip()
        if not user:
            continue
        if user.lower() in {"стоп", "stop"}:
            print("[Interviewer]: Ок, остановимся. Спасибо!")
            state['done_info'] = 1
            state = graph.invoke(state)
            print(state)
            break

        state["history"].append(HumanMessage(content=user))

        # прогоняем граф
        state = graph.invoke(state)

        # показываем скрытый совет (можешь убрать, если не нужно)
        print(f"\n--- tech (hidden) ---\n{state['tech_advice']}\n--------------------\n")
        print(state["done_info"])
        print(f"\n--- control (hidden) ---\n{state['control_advice']}\n--------------------\n")
        print(f"\n--- control (hidden) ---\n{state['candidate_info']}\n--------------------\n")
        print(f"[Interviewer]: {state['interviewer_msg']}")


if __name__ == "__main__":
    main()
