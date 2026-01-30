
# Multi-Agent Interview CLI

## 1) Агенты системы
- **Control** — заполняет карточку кандидата (name / role / grade / exp) из диалога и подсказывает, что спросить, чтобы добрать пустые поля.
- **Tech** — формирует/обновляет план тем и предлагает следующий *один* технический вопрос (с адаптацией сложности).
- **Interviewer** — превращает советы Control/Tech в короткое, “человеческое” сообщение кандидату (1–3 предложения).
- **Observer** — на завершении формирует финальный отчёт (**final_feedback**) по диалогу.

## 2) Логика взаимодействия
1) Интервьюер задаёт вопрос → кандидат отвечает в CLI.  
2) Пока карточка кандидата не заполнена — работает **Control** (добираем name/role/grade/exp).  
3) Когда карточка заполнена — включается **Tech** (техвопросы по плану).  
4) **Interviewer** на каждом ходе пишет следующее сообщение кандидату.  
5) На `stop/стоп` → запускается **Observer**, создаётся финальный отчёт и сохраняются файлы:
   - `interview_log_{scenario_id}.json`
   - `final_feedback_{scenario_id}.md`

## 3) Пример работы
```text
scenario_id (1..5): 1
Ваше ФИО (participant_name): Иванов Иван
[Interviewer]: Привет! Расскажи о себе.
[You]: Я Python backend, 3 года, работал с FastAPI и Postgres.
[Interviewer]: Ок, а на каком грейде себя видишь и почему?
[You]: Middle.
...
[You]: stop
[Interviewer]: Спасибо! Ниже финальный фидбек:
(созданы interview_log_1.json и final_feedback_1.md)
```

## 4) Что нужно для запуска

* Python **3.10+**
* ключ **Mistral** в переменной окружения `MISTRAL_API_KEY`
* зависимости из `requirements.txt`

## 5) Как запустить

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Создай `.env`:

```env
MISTRAL_API_KEY=ваш_ключ
```

Запуск:

```bash
python main.py
```


