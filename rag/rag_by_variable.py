import os
import re
import json
import time
import math
import hashlib
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import requests

import concurrent.futures
import textwrap
import threading
warnings.filterwarnings("ignore")

SOURCES_DIR = os.environ.get(
    "PARSED_SOURCES_DIR",
    "/home/ubuntu/esg-pipeline/data/parsed_sources"
)

OUTPUT_CSV = os.environ.get(
    "RAG_OUTPUT_CSV",
    "/home/ubuntu/esg-pipeline/data/output/rag_by_variable.csv"
)

INTERMEDIATE_DIR = os.environ.get(
    "RAG_INTERMEDIATE_DIR",
    "/home/ubuntu/esg-pipeline/data/intermediate"
)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8080")
CLIENT_API_KEY = os.environ.get("CLIENT_API_KEY", "")
GATEWAY_ROLE = os.environ.get("RAG_GATEWAY_ROLE", "scorer")
GATEWAY_TIMEOUT = int(os.environ.get("RAG_LLM_TIMEOUT", "300"))

RAG_MAX_COMPANIES = os.environ.get("RAG_MAX_COMPANIES", "").strip()
RAG_COMPANY_FILTER = os.environ.get("RAG_COMPANY_FILTER", "").strip()
RAG_LOW_COVERAGE_CHARS = int(os.environ.get("RAG_LOW_COVERAGE_CHARS", "10")) #имнимальное количество символов в доке для работоспособности
RAG_DOC_DEDUPE_THRESHOLD = float(os.environ.get("RAG_DOC_DEDUPE_THRESHOLD", "0.93")) # граница, меньше которой доки считаются разными
RAG_DOC_DEDUPE_DATE_TOLERANCE_DAYS = int(os.environ.get("RAG_DOC_DEDUPE_DATE_TOLERANCE_DAYS", "3")) #количество дней недопустимых для 

# ============================================================
# Настройки дедупликации документов
# ============================================================

# Для точного дубля используется hash нормализованного текста.

# Порог для практически идентичных документов.
# Намеренно высокий, чтобы не спутать отчёты разных лет.
RAG_DOC_NEAR_DUP_THRESHOLD = float(
    os.environ.get("RAG_DOC_NEAR_DUP_THRESHOLD", "0.999")
)

# Минимальное отношение длины текстов для near-duplicate.
RAG_DOC_NEAR_DUP_LENGTH_RATIO = float(
    os.environ.get("RAG_DOC_NEAR_DUP_LENGTH_RATIO", "0.98")
)

# Сколько символов сравнивать в начале и конце текста.
RAG_DOC_DUP_EDGE_CHARS = int(
    os.environ.get("RAG_DOC_DUP_EDGE_CHARS", "1000")
)

Path(INTERMEDIATE_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(OUTPUT_CSV) or ".").mkdir(parents=True, exist_ok=True)

HEADER_COLS = [
    "Company_name",
    "Domains",
    "Variable",
    "Evidence_Type",
    "Evidence_Type_Explanation",
    "Evidence_Citation",
    "Evidence_Link",
    "Evidence_Date",
    "Freshness",
    "Score",
    "Notes",
    "LLM",
    "Evaluation_Date",
    "INN",
]

VARIABLES = [
    "E1", "E2", "E3", "E4", "E5", "E6", "E7",
    "Sc1", "Sc2", "Sc3", "Sc4",
    "Se1", "Se2", "Se3", "Se4", "Se5",
    "Ss1", "Ss2", "Ss3", "Ss4",
    "So1", "So2", "So3", "So4",
    "G1", "G2", "G3", "G4", "G5", "G6",
]

ESG_QUERIES = {
    "E1":  [
        "экологическая политика", "экологический менеджмент", "ISO 14001",
            "environmental policy", "environmental management system", "environmental monitoring",
            "keskkonnapoliitika", "keskkonnajuhtimine", "keskkonnajuhtimissüsteem",
            "keskkonnajuhtimissusteem", "keskkonnakontroll", "keskkonnaseire",
            "keskkonnaluba", "keskkonnaeesmärk", "keskkonnaeesmark"
    ],
    "E2":  [
        "выбросы", "сбросы", "загрязнение", "очистка сточных вод", "экологические инциденты",
            "pollution", "emissions", "effluent", "discharge", "wastewater treatment",
            "heitmed", "heitkogused", "reovesi", "heitvesi", "reoveepuhastus", "veepuhastus",
            "saaste", "saastamine", "reostus", "keskkonnareostus", "NOx", "SOx"
    ],
    "E3":  [
        "парниковые газы", "CO2", "CO2e", "Scope 1", "Scope 2", "Scope 3",
            "углеродный след", "декарбонизация", "климатические риски",
            "greenhouse gas", "GHG", "carbon footprint", "climate strategy", "climate risk",
            "net zero", "decarbonisation", "decarbonization",
            "kasvuhoonegaas", "süsinikuheide", "susinikuheide", "süsinikujalajälg",
            "susinikujalajalg", "kliimarisk", "kliimaneutraal", "dekarboniseer"
    ],
    "E4": [
           "отходы", "опасные отходы", "раздельный сбор", "переработка", "утилизация",
        "осадок сточных вод", "зола", "шлам",
        "waste", "hazardous waste", "recycling", "reuse", "sludge", "ash",
        "jäätmed", "jaatmed", "jäätmekäitlus", "jaatmekaitlus", "jäätmete sortimine",
        "ohtlikud jäätmed", "ringlussevõtt", "taaskasutus", "reoveesete", "setted", "tuhk"
        ],
    "E5": [
             "энергопотребление", "энергоэффективность", "энергосбережение", "сетевые потери",
        "тепловые потери", "энергоаудит", "ISO 50001", "возобновляемая энергия",
        "energy consumption", "energy efficiency", "network losses", "heat loss",
        "energy audit", "renewable energy", "fuel consumption",
        "energiatarbimine", "energiaefektiivsus", "energiasääst", "energiaaudit",
        "võrgukadu", "vorgukadu", "soojuskadu", "kütusekulu", "kutusekulu",
        "taastuvenergia", "elektrienergia tarbimine", "soojusenergia", "päikeseenergia",
        # универсальные
        "энергоменеджмент", "потребление электроэнергии", "потребление топлива",
    "энергосберегающие технологии", "удельный расход энергии", "энергетический паспорт",
    "энергоэффективное оборудование",
    "energy management", "electricity consumption", "energy-saving technology",
    "energy performance", "power consumption"
        ],
    "E6": [
            "водозабор", "водопотребление", "водные ресурсы", "потери воды", "утечки воды",
        "повторное использование воды",
        "water abstraction", "water consumption", "water resources", "water loss", "leakage",
        "veevõtt", "veevott", "veekasutus", "veetarbimine", "veekadu", "veeleke",
        "lekked", "veevaru", "põhjavesi", "pohjavesi", "pinnavesi",
        # + универсальные добавки
    "расход воды", "экономия воды", "водосбережение", "оборотная вода",
    "очистка сточных вод", "сброс сточных вод",
    "water saving", "water efficiency", "wastewater treatment", "water reuse", "water footprint"
        ],
    "E7": [
             "биоразнообразие", "экосистемы", "рекультивация", "охраняемые природные территории",
        "biodiversity", "ecosystem", "protected area", "habitat", "restoration",
        "environmental impact assessment",
        "elurikkus", "ökosüsteem", "okosusteem", "looduskaitse", "kaitseala",
        "elupaik", "rekultiveerimine", "keskkonnamõju hindamine", "keskkonnamoju hindamine"
        ],

    "Sc1": [
        "ISO 9001", "система менеджмента качества", "качество услуг", "контроль качества",
            "качество питьевой воды", "service quality", "quality management", "quality control",
            "water quality", "laboratory control", "continuity of service",
            "kvaliteedijuhtimine", "kvaliteedikontroll", "teenuse kvaliteet", "teenusekvaliteet",
            "kvaliteedinõuded", "kvaliteedinouded", "veekvaliteet", "vee kvaliteet",
            "laboratoorne kontroll", "varustuskindlus", "tarnekindlus", "töökindlus",
            # + универсальные добавки
    "контроль качества продукции", "стандарты качества", "сертификат соответствия",
    "ГОСТ", "приёмка работ", "входной контроль", "система контроля качества",
    "quality assurance", "quality standards", "certificate of conformity",
    "product quality control", "quality inspection"
    ],
    "Sc2": [
        "безопасность услуг", "безопасность питьевой воды", "санитарный контроль",
            "обеззараживание", "хлорирование", "UV", "аварийные процедуры",
            "service safety", "drinking water safety", "water safety", "disinfection",
            "chlorination", "UV treatment", "emergency procedure", "security of supply",
            "teenuseohutus", "joogivee ohutus", "joogivesi", "veeproov", "veeanalüüs",
            "veeanaluus", "labor", "desinfitseerimine", "kloorimine", "UV-töötlus",
            "tehnilised tingimused", "liitumistingimused", "ohutusnõuded", "ohutusnouded",
            "katkestus", "avarii", "rikked", "varustuskindlus",
            # + универсальные добавки
    "безопасность продукции", "требования безопасности", "сертификат безопасности",
    "техническая безопасность", "инцидент безопасности", "процедура отзыва продукции",
    "product safety", "safety standards", "safety requirements", "recall procedure"
    ],
    "Sc3": [
            "политика конфиденциальности", "персональные данные", "обработка персональных данных",
        "информирование клиентов", "условия оказания услуг",
        "privacy policy", "personal data", "data protection", "customer information",
        "terms of service", "service conditions", "tariff information",
        "privaatsus", "privaatsuspoliitika", "isikuandmed", "andmekaitse",
        "andmetöötlus", "andmetootlus", "kliendiandmed", "kliendi teavitamine",
        "teenustingimused", "tüüptingimused", "tuuptingimused", "hinnakiri"
        ],
    "Sc4": [
           "жалобы клиентов", "претензии", "обращения клиентов", "обратная связь",
        "порядок рассмотрения обращений", "клиентский сервис",
        "complaints", "claims", "customer feedback", "customer service", "complaint procedure",
        "kaebus", "kaebused", "pretensioon", "pretensioonid", "tagasiside",
        "pöördumine", "poordumine", "klienditeenindus", "klienditugi",
        "avariiteade", "rikketeade", "vastamise tähtaeg", "vastamise tahtaeg"
        ],
    "Se1": [
       "охрана труда", "безопасность труда", "ISO 45001", "травматизм", "несчастные случаи",
           "occupational safety", "occupational health", "work safety", "workplace risk assessment",
           "tööohutus", "tooohutus", "töötervishoid", "tootervishoid", "töökeskkond",
           "tookeskkond", "tööõnnetus", "tooonnetus", "riskianalüüs", "riskianaluus",
           "juhendamine", "kaitsevahendid"
    ],
    "Se2": [
        "условия труда", "оплата труда", "социальный пакет", "коллективный договор",
            "расходы на персонал", "employee benefits", "working conditions",
            "collective agreement", "personnel costs", "staff costs",
            "töötingimused", "tootingimused", "töötasu", "tootasu", "palgakulu",
            "personalikulu", "tööjõukulud", "toojoukulud", "hüved", "huved", "kollektiivleping"
    ],
    "Se3": [
            "обучение персонала", "повышение квалификации", "профессиональное развитие",
        "расходы на обучение", "training", "professional development", "training costs",
        "employee development", "qualification",
        "koolitus", "koolitused", "koolituskulud", "täienduskoolitus", "taienduskoolitus",
        "väljaõpe", "valjaope", "kutsekvalifikatsioon", "pädevus", "padevus"
        ],
    "Se4": [
            "равные возможности", "недискриминация", "гендерное равенство", "доля женщин",
        "diversity", "inclusion", "equal opportunity", "non-discrimination", "gender equality",
        "võrdsed võimalused", "vordsed voimalused", "mittediskrimineerimine",
        "sooline võrdõiguslikkus", "sooline vordoiguslikkus", "naiste osakaal", "kaasamine"
        ],
    "Se5": [
            "вовлечённость сотрудников", "опрос сотрудников", "внутренние коммуникации",
        "профсоюз", "совет работников", "employee engagement", "employee survey",
        "trade union", "employee representative",
        "töötajate kaasamine", "tootajate kaasamine", "töötajate rahulolu",
        "tootajate rahulolu", "töötajate küsitlus", "tootajate kusitlus",
        "ametiühing", "ametiuhing", "töötajate esindaja"
        ],
    "Ss1": [
        "кодекс поставщика", "требования к поставщикам", "ответственные закупки",
            "supplier code", "supplier requirements", "responsible procurement", "responsible sourcing",
            "tarnijakoodeks", "tarnijate nõuded", "hankekord", "hankepoliitika",
            "hanketingimused", "vastutustundlikud hanked"
    ],
    "Ss2": [
        "проверка поставщиков", "проверка контрагентов", "due diligence", "аудит поставщиков",
            "supplier assessment", "supplier audit", "supplier due diligence", "counterparty screening",
            "tarnijate hindamine", "tarnijate audit", "partnerite kontroll",
            "vastaspoolte kontroll", "taustakontroll", "hoolsuskohustus"
    ],
     "Ss3": [
            "положение о закупках", "правила закупок", "конкурентные закупки", "равный доступ",
        "procurement rules", "competitive procurement", "public procurement", "tender procedure",
        "hankekord", "hankemenetlus", "riigihange", "riigihanked",
        "pakkumiskutse", "konkurents", "võrdne kohtlemine", "vordne kohtlemine"
        ],
    "Ss4": [
           "локальные поставщики", "поддержка МСП", "развитие поставщиков",
        "local suppliers", "SME suppliers", "supplier development", "supplier training",
        "kohalikud tarnijad", "kohalikud pakkujad", "väikeettevõtted", "vaikeettevotted",
        "VKE", "tarnijate arendamine", "tarnijate koolitus"
        ],
    "So1": [
         "благотворительность", "социальные проекты", "социальные инвестиции", "волонтёрство",
            "спонсорство", "community investment", "charity", "donation", "sponsorship", "volunteering",
            "heategevus", "sotsiaalprojekt", "sotsiaalsed projektid", "annetus", "sponsorlus",
            "vabatahtlik tegevus", "hariduse toetamine", "spordi toetamine", "kultuuri toetamine"
    ],
    "So2": [
        "общественные слушания", "встречи с жителями", "диалог с сообществами",
            "публичные консультации", "соглашения с муниципалитетами",
            "community dialogue", "public consultation", "public hearing", "resident meeting",
            "avalik arutelu", "avalik konsultatsioon", "elanike koosolek", "kogukonna dialoog",
            "kohalik omavalitsus", "omavalitsus", "linnavalitsus", "vallavalitsus"
    ],
    "So3": [
           "НКО", "отраслевые ассоциации", "профессиональные объединения", "экспертный совет",
        "университет", "NGO", "industry association", "professional association",
        "expert council", "university partnership", "stakeholder working group",
        "vabaühendus", "vabauhendus", "erialaliit", "kutseliit", "ettevõtjate liit",
        "ettevotjate liit", "ülikool", "ulikool", "eksperdikogu", "töörühm", "tooruhm", "liikmelisus"
        ],
    "So4": [
            "воздействие на местные сообщества", "шум", "запах", "дорожные работы",
        "земляные работы", "компенсации жителям",
        "community impact", "noise", "odour", "odor", "traffic impact", "construction impact",
        "kogukonnamõju", "kogukonnamoju", "müra", "mura", "lõhn", "lohn",
        "kaevetööd", "kaevetood", "teetööd", "teetood", "elanike kaebused", "hüvitis", "huvitis",
        # + универсальные добавки
    "ограничение движения", "нарушение благоустройства", "строительные работы рядом",
    "воздействие на инфраструктуру района",
    "local infrastructure impact", "construction disturbance", "road closure"
        ],
    "G1":  [
        "ESG стратегия", "стратегия устойчивого развития", "долгосрочные цели",
            "миссия", "приоритеты развития", "strategy", "sustainability strategy",
            "long-term objectives", "mission", "development plan",
            "strateegia", "arengukava", "arenguplaan", "missioon", "eesmärgid", "eesmargid",
            "prioriteedid", "kestlikkus", "jätkusuutlikkus", "varustuskindlus", "teenuse kvaliteet"
    ],
    "G2":  [
        "кодекс этики", "антикоррупционная политика", "конфликт интересов", "комплаенс",
            "code of ethics", "anti-corruption", "conflict of interest", "compliance", "code of conduct",
            "eetikakoodeks", "eetika", "korruptsioonivastane", "korruptsioon",
            "huvide konflikt", "vastavus", "käitumiskoodeks", "kaitumiskoodeks"
    ],
    "G3":  [
        "горячая линия", "линия доверия", "whistleblowing", "сообщение о нарушении",
            "анонимное сообщение", "защита заявителя", "speak up", "report misconduct",
            "whistleblower protection", "vihjeliin", "usaldustelefon", "rikkumisest teavitamine",
            "rikkumisteade", "anonüümne teade", "anonuumne teade", "teavitaja kaitse"
    ],
    "G4": [
             "совет директоров", "наблюдательный совет", "правление", "корпоративное управление",
        "комитет", "board", "supervisory board", "management board", "governance",
        "nõukogu", "noukogu", "juhatus", "juhtimine", "ühingujuhtimine", "uhingujuhtimine",
        "komitee", "auditikomitee", "juhtimisstruktuur", "vastutus"
        ],
    "G5": [
           "годовой отчёт", "годовой отчет", "финансовая отчётность", "раскрытие информации",
        "структура собственности", "annual report", "financial statements", "disclosure",
        "ownership", "shareholders", "report archive",
        "majandusaasta aruanne", "majandusaasta aruanded", "aastaaruanne", "aastaaruanded",
        "finantsaruanne", "raamatupidamise aastaaruanne", "aruanded", "aruandlus",
        "teabe avalikustamine", "omanik", "omanikud", "aktsionär", "aktsionar"
        ],
    "G6": [
           "управление рисками", "внутренний контроль", "внутренний аудит", "реестр рисков",
        "оценка рисков", "risk management", "internal control", "internal audit",
        "risk assessment", "risk register", "audit committee",
        "riskijuhtimine", "sisekontroll", "siseaudit", "riskihindamine",
        "riskiregister", "auditikomitee", "kontrollisüsteem", "kontrollisusteem"
        ],
}

UNIVERSAL_NEGATIVE_TERMS = {

    # -------------------------------------------------------------------------
    # СОЦИАЛЬНЫЕ / БЛАГОТВОРИТЕЛЬНЫЕ ПРОЕКТЫ
    # -------------------------------------------------------------------------
    "charity": [
        # RU
        "благотворитель",
        "благотворительность",
        "благотворительный",
        "пожертвование",
        "пожертвования",
        "пожертвовать",
        "благотворительный взнос",
        "благотворительная помощь",
        "благотворительная деятельность",
        "благотворительный проект",
        "благотворительная программа",
        "помощь благотворительным",
        "помощь фонду",
        "поддержка фонда",

        # EN
        "charity",
        "charitable",
        "charitable activity",
        "charitable activities",
        "charitable contribution",
        "charitable contributions",
        "charitable donation",
        "charitable donations",
        "donation",
        "donations",
        "donate",
        "donated",
        "donor",
        "donors",
        "charity project",
        "charity program",
        "support for charities",
        "support for a charity",
        "support for foundations",
    ],

    # -------------------------------------------------------------------------
    # СПОНСОРСТВО
    # -------------------------------------------------------------------------
    "sponsorship": [
        # RU
        "спонсор",
        "спонсоры",
        "спонсорство",
        "спонсорский",
        "спонсорская поддержка",
        "спонсорский вклад",
        "спонсорская помощь",
        "спонсорский проект",

        # EN
        "sponsor",
        "sponsors",
        "sponsorship",
        "sponsoring",
        "sponsored",
        "sponsorship program",
        "sponsorship support",
        "sponsorship contribution",
    ],

    # -------------------------------------------------------------------------
    # ВОЛОНТЕРСТВО
    # -------------------------------------------------------------------------
    "volunteering": [
        # RU
        "волонтер",
        "волонтеры",
        "волонтерство",
        "волонтерская деятельность",
        "волонтерская программа",
        "волонтерские часы",
        "волонтерская помощь",

        # EN
        "volunteer",
        "volunteers",
        "volunteering",
        "voluntary work",
        "volunteer activity",
        "volunteer activities",
        "volunteer program",
        "volunteer hours",
        "employee volunteering",
    ],

    # -------------------------------------------------------------------------
    # ВЗАИМОДЕЙСТВИЕ С МЕСТНЫМИ СООБЩЕСТВАМИ
    # -------------------------------------------------------------------------
    "community_dialogue": [
        # RU
        "общественные слушания",
        "публичные слушания",
        "встречи с жителями",
        "встреча с жителями",
        "диалог с сообществом",
        "диалог с сообществами",
        "взаимодействие с местным сообществом",
        "взаимодействие с сообществами",
        "консультации с населением",
        "общественные консультации",
        "публичные консультации",
        "обсуждение с жителями",
        "обсуждения с жителями",
        "обращения жителей",
        "жалобы жителей",
        "мнение жителей",
        "мнение местного сообщества",
        "местное сообщество",
        "местные сообщества",

        # EN
        "community dialogue",
        "community engagement",
        "community consultation",
        "community consultations",
        "public consultation",
        "public consultations",
        "public hearing",
        "public hearings",
        "resident meeting",
        "resident meetings",
        "meetings with residents",
        "local community",
        "local communities",
        "community concerns",
        "community complaints",
        "community feedback",
        "resident feedback",
        "stakeholder consultation",
    ],

    # -------------------------------------------------------------------------
    # ВОЗДЕЙСТВИЕ НА МЕСТНОЕ СООБЩЕСТВО / ИНФРАСТРУКТУРУ
    # -------------------------------------------------------------------------
    "community_impact": [
        # RU
        "воздействие на местные сообщества",
        "воздействие на местное сообщество",
        "воздействие на жителей",
        "влияние на жителей",
        "воздействие на население",
        "влияние на население",
        "нарушение благоустройства",
        "ограничение движения",
        "перекрытие дороги",
        "дорожные работы",
        "строительные работы",
        "земляные работы",
        "шум",
        "шумовое воздействие",
        "запах",
        "неприятный запах",
        "пылевое воздействие",
        "вибрация",
        "компенсация жителям",
        "компенсации жителям",
        "ущерб местному сообществу",
        "влияние на инфраструктуру",
        "воздействие на инфраструктуру",

        # EN
        "community impact",
        "impact on local communities",
        "impact on local community",
        "impact on residents",
        "impact on local residents",
        "impact on population",
        "local infrastructure impact",
        "infrastructure impact",
        "traffic impact",
        "traffic disruption",
        "road works",
        "road closure",
        "construction works",
        "construction impact",
        "construction disturbance",
        "excavation works",
        "noise",
        "noise impact",
        "odour",
        "odor",
        "dust impact",
        "vibration",
        "compensation to residents",
        "resident compensation",
        "community damage",
    ],

    # -------------------------------------------------------------------------
    # ПОСТАВЩИКИ / ОТВЕТСТВЕННЫЕ ЗАКУПКИ
    # -------------------------------------------------------------------------
    "supplier_policy": [
        # RU
        "кодекс поставщика",
        "кодекс поведения поставщика",
        "требования к поставщикам",
        "требования для поставщиков",
        "ответственные закупки",
        "ответственный закупочный процесс",
        "ответственный выбор поставщиков",
        "политика в отношении поставщиков",
        "политика работы с поставщиками",
        "принципы работы с поставщиками",
        "стандарты для поставщиков",
        "требования к контрагентам",

        # EN
        "supplier code",
        "supplier code of conduct",
        "supplier requirements",
        "requirements for suppliers",
        "responsible procurement",
        "responsible sourcing",
        "responsible purchasing",
        "supplier policy",
        "supplier policies",
        "supplier standards",
        "supplier principles",
        "supplier expectations",
        "requirements for suppliers",
        "supplier conduct",
    ],

    # -------------------------------------------------------------------------
    # ПРОВЕРКА / ОЦЕНКА ПОСТАВЩИКОВ
    # -------------------------------------------------------------------------
    "supplier_due_diligence": [
        # RU
        "проверка поставщиков",
        "оценка поставщиков",
        "аудит поставщиков",
        "проверка контрагентов",
        "оценка контрагентов",
        "проверка благонадежности поставщиков",
        "проверка благонадежности контрагентов",
        "комплексная проверка поставщиков",
        "комплексная проверка контрагентов",
        "надлежащая проверка",
        "должная проверка",
        "анализ поставщиков",
        "мониторинг поставщиков",

        # EN
        "supplier assessment",
        "supplier assessments",
        "supplier evaluation",
        "supplier audit",
        "supplier audits",
        "supplier due diligence",
        "supplier screening",
        "supplier monitoring",
        "supplier review",
        "supplier reviews",
        "counterparty screening",
        "counterparty due diligence",
        "due diligence",
        "supplier compliance assessment",
    ],

    # -------------------------------------------------------------------------
    # ПРОЦЕДУРЫ ЗАКУПОК / ТЕНДЕРЫ
    # -------------------------------------------------------------------------
    "procurement_process": [
        # RU
        "положение о закупках",
        "правила закупок",
        "процедура закупок",
        "процедуры закупок",
        "закупочная процедура",
        "закупочные процедуры",
        "конкурентные закупки",
        "конкурентная процедура",
        "тендерная процедура",
        "тендер",
        "тендеры",
        "конкурсная процедура",
        "равный доступ к закупкам",
        "прозрачность закупок",

        # EN
        "procurement rules",
        "procurement procedure",
        "procurement procedures",
        "procurement process",
        "procurement processes",
        "competitive procurement",
        "competitive bidding",
        "tender procedure",
        "tender procedures",
        "tender",
        "tenders",
        "public procurement",
        "procurement transparency",
        "equal access to procurement",
    ],

    # -------------------------------------------------------------------------
    # ЛОКАЛЬНЫЕ ПОСТАВЩИКИ / МСП
    # -------------------------------------------------------------------------
    "local_suppliers": [
        # RU
        "локальные поставщики",
        "местные поставщики",
        "местный поставщик",
        "поддержка местных поставщиков",
        "развитие поставщиков",
        "развитие местных поставщиков",
        "поддержка МСП",
        "малый и средний бизнес",
        "малые и средние предприятия",
        "местный бизнес",
        "местные предприятия",

        # EN
        "local suppliers",
        "local supplier",
        "support for local suppliers",
        "local supplier development",
        "supplier development",
        "SME suppliers",
        "small and medium-sized enterprises",
        "small and medium enterprises",
        "small and medium-sized businesses",
        "local businesses",
        "local enterprises",
    ],

    # -------------------------------------------------------------------------
    # ЭТИКА / АНТИКОРРУПЦИЯ
    # -------------------------------------------------------------------------
    "ethics_anticorruption": [
        # RU
        "кодекс этики",
        "этический кодекс",
        "кодекс поведения",
        "антикоррупционная политика",
        "антикоррупционные меры",
        "борьба с коррупцией",
        "противодействие коррупции",
        "предотвращение коррупции",
        "взяточничество",
        "взятка",
        "взятки",
        "конфликт интересов",
        "конфликты интересов",
        "комплаенс",
        "комплаенс программа",
        "этические нормы",
        "деловая этика",

        # EN
        "code of ethics",
        "ethics code",
        "code of conduct",
        "anti-corruption policy",
        "anti-corruption measures",
        "anti-corruption",
        "anti bribery",
        "anti-bribery",
        "corruption prevention",
        "bribery",
        "bribe",
        "bribes",
        "conflict of interest",
        "conflicts of interest",
        "compliance",
        "compliance program",
        "business ethics",
        "ethical standards",
    ],

    # -------------------------------------------------------------------------
    # WHISTLEBLOWING / СООБЩЕНИЯ О НАРУШЕНИЯХ
    # -------------------------------------------------------------------------
    "whistleblowing": [
        # RU
        "горячая линия",
        "линия доверия",
        "линия для сообщений",
        "сообщение о нарушении",
        "сообщения о нарушениях",
        "сообщить о нарушении",
        "сообщение о неправомерных действиях",
        "анонимное сообщение",
        "анонимные сообщения",
        "анонимная жалоба",
        "защита заявителя",
        "защита информатора",
        "защита лиц сообщивших",
        "информатор",
        "заявитель",
        "раскрытие нарушений",

        # EN
        "whistleblowing",
        "whistleblower",
        "whistleblowers",
        "whistleblower protection",
        "hotline",
        "ethics hotline",
        "compliance hotline",
        "report misconduct",
        "reporting misconduct",
        "report a violation",
        "report violations",
        "anonymous reporting",
        "anonymous report",
        "speak up",
        "speak-up",
        "protected disclosure",
        "reporting channel",
        "reporting channels",
    ],

    # -------------------------------------------------------------------------
    # СТРАТЕГИЯ / УСТОЙЧИВОЕ РАЗВИТИЕ
    # -------------------------------------------------------------------------
    "strategy": [
        # RU
        "ESG стратегия",
        "стратегия устойчивого развития",
        "стратегия устойчивости",
        "стратегические цели",
        "долгосрочные цели",
        "долгосрочная стратегия",
        "стратегический план",
        "план развития",
        "приоритеты развития",
        "миссия компании",
        "стратегические приоритеты",

        # EN
        "ESG strategy",
        "sustainability strategy",
        "sustainability plan",
        "sustainability goals",
        "strategic goals",
        "long-term objectives",
        "long-term strategy",
        "strategic plan",
        "development plan",
        "strategic priorities",
        "company mission",
    ],

    # -------------------------------------------------------------------------
    # КОРПОРАТИВНОЕ УПРАВЛЕНИЕ
    # -------------------------------------------------------------------------
    "governance": [
        # RU
        "совет директоров",
        "наблюдательный совет",
        "правление",
        "корпоративное управление",
        "система корпоративного управления",
        "структура управления",
        "управленческая структура",
        "комитет совета директоров",
        "аудиторский комитет",
        "комитет по аудиту",
        "ответственность совета директоров",
        "полномочия совета директоров",

        # EN
        "board of directors",
        "board",
        "supervisory board",
        "management board",
        "corporate governance",
        "governance structure",
        "management structure",
        "board committee",
        "audit committee",
        "board responsibilities",
        "board responsibility",
        "board powers",
    ],

    # -------------------------------------------------------------------------
    # ОТЧЕТНОСТЬ / РАСКРЫТИЕ / СОБСТВЕННОСТЬ
    # -------------------------------------------------------------------------
    "reporting_disclosure": [
        # RU
        "годовой отчет",
        "годовой отчёт",
        "годовая отчетность",
        "годовая отчётность",
        "финансовая отчетность",
        "финансовая отчётность",
        "раскрытие информации",
        "раскрытие данных",
        "публичное раскрытие",
        "структура собственности",
        "акционеры",
        "структура акционеров",
        "отчетность компании",
        "корпоративная отчетность",

        # EN
        "annual report",
        "annual reporting",
        "financial statements",
        "financial reporting",
        "disclosure",
        "information disclosure",
        "public disclosure",
        "ownership structure",
        "shareholders",
        "shareholder structure",
        "company reporting",
        "corporate reporting",
    ],

    # -------------------------------------------------------------------------
    # РИСКИ / ВНУТРЕННИЙ КОНТРОЛЬ / АУДИТ
    # -------------------------------------------------------------------------
    "risk_control": [
        # RU
        "управление рисками",
        "оценка рисков",
        "реестр рисков",
        "карта рисков",
        "внутренний контроль",
        "система внутреннего контроля",
        "внутренний аудит",
        "внутренний аудитор",
        "комитет по рискам",
        "контроль рисков",
        "управление рисками компании",

        # EN
        "risk management",
        "risk assessment",
        "risk register",
        "risk map",
        "internal control",
        "internal controls",
        "internal control system",
        "internal audit",
        "internal auditor",
        "risk committee",
        "risk control",
        "enterprise risk management",
    ],

    # -------------------------------------------------------------------------
    # ВОВЛЕЧЕННОСТЬ СОТРУДНИКОВ
    # -------------------------------------------------------------------------
    "employee_engagement": [
        # RU
        "вовлеченность сотрудников",
        "вовлечённость сотрудников",
        "опрос сотрудников",
        "опрос работников",
        "удовлетворенность сотрудников",
        "удовлетворённость сотрудников",
        "удовлетворенность работников",
        "внутренние коммуникации",
        "обратная связь сотрудников",
        "обратная связь работников",
        "профсоюз",
        "профсоюзы",
        "совет работников",
        "представитель работников",

        # EN
        "employee engagement",
        "employee survey",
        "employee surveys",
        "employee satisfaction",
        "employee feedback",
        "workforce engagement",
        "internal communications",
        "employee communication",
        "trade union",
        "trade unions",
        "labor union",
        "labour union",
        "employee representative",
        "employee representatives",
    ],

    # -------------------------------------------------------------------------
    # РАЗНООБРАЗИЕ / РАВНЫЕ ВОЗМОЖНОСТИ
    # -------------------------------------------------------------------------
    "diversity_inclusion": [
        # RU
        "разнообразие",
        "инклюзивность",
        "инклюзия",
        "равные возможности",
        "равенство возможностей",
        "недискриминация",
        "дискриминация",
        "гендерное равенство",
        "гендерный баланс",
        "женщины в руководстве",
        "доля женщин",
        "вовлечение",

        # EN
        "diversity",
        "inclusion",
        "inclusiveness",
        "equal opportunity",
        "equal opportunities",
        "non-discrimination",
        "non discrimination",
        "discrimination",
        "gender equality",
        "gender balance",
        "women in leadership",
        "share of women",
        "female representation",
    ],
}
# =============================================================================
# КАКИЕ НЕГАТИВНЫЕ КАТЕГОРИИ МОГУТ БЫТЬ ПУТАНИЦЕЙ ДЛЯ КАЖДОЙ ПЕРЕМЕННОЙ
#
# Здесь указываются только действительно близкие темы.
# Категории, относящиеся непосредственно к самой переменной, НЕ штрафуются.
# =============================================================================

NEGATIVE_CATEGORIES_BY_VAR = {

    # -------------------------------------------------------------------------
    # Supply chain
    # -------------------------------------------------------------------------
    "Ss1": [
        "supplier_due_diligence",
        "procurement_process",
        "local_suppliers",
    ],

    "Ss2": [
        "supplier_policy",
        "procurement_process",
        "local_suppliers",
    ],

    "Ss3": [
        "supplier_policy",
        "supplier_due_diligence",
        "local_suppliers",
    ],

    "Ss4": [
        "supplier_policy",
        "supplier_due_diligence",
        "procurement_process",
    ],

    # -------------------------------------------------------------------------
    # Society / community
    # -------------------------------------------------------------------------
    "So1": [
        "community_dialogue",
        "community_impact",
    ],

    "So2": [
        "charity",
        "sponsorship",
        "volunteering",
        "community_impact",
    ],

    "So3": [
        "charity",
        "sponsorship",
        "volunteering",
        "community_dialogue",
    ],

    "So4": [
        "charity",
        "sponsorship",
        "volunteering",
        "community_dialogue",
    ],

    # -------------------------------------------------------------------------
    # Governance
    # -------------------------------------------------------------------------
    "G1": [
        "ethics_anticorruption",
        "whistleblowing",
        "governance",
        "reporting_disclosure",
        "risk_control",
    ],

    "G2": [
        "whistleblowing",
        "governance",
        "strategy",
        "reporting_disclosure",
        "risk_control",
    ],

    "G3": [
        "ethics_anticorruption",
        "governance",
        "strategy",
        "reporting_disclosure",
        "risk_control",
    ],

    "G4": [
        "ethics_anticorruption",
        "whistleblowing",
        "strategy",
        "reporting_disclosure",
        "risk_control",
    ],

    "G5": [
        "strategy",
        "governance",
        "ethics_anticorruption",
        "whistleblowing",
        "risk_control",
    ],

    "G6": [
        "strategy",
        "ethics_anticorruption",
        "whistleblowing",
        "governance",
        "reporting_disclosure",
    ],
}

# =============================================================================
# ДВА ЭТАПА ОЦЕНКИ — 30 переменных
# =============================================================================
STAGE1_VARIABLES = [
    "E1", "E2", "E3", "E4", "E5", "E6", "E7",
    "Sc1", "Sc2", "Sc3", "Sc4",
    "Se1", "Se2", "Se3", "Se4", "Se5",
]

STAGE2_VARIABLES = [
    "Ss1", "Ss2", "Ss3", "Ss4",
    "So1", "So2", "So3", "So4",
    "G1", "G2", "G3", "G4", "G5", "G6",
]

STAGE1_QUERIES = {v: ESG_QUERIES[v] for v in STAGE1_VARIABLES}
STAGE2_QUERIES = {v: ESG_QUERIES[v] for v in STAGE2_VARIABLES}

EVIDENCE_MARKERS = {
    "certificate": ["сертификат", "сертификац", "iso", "breeam", "leed", "лицензия"],
    "KPI": ["%", "тонн", "м3", "квт", "показател", "количество", "объем", "объём", "снижение", "рост"],
    "target": ["цель", "к 20", "до 20", "снизить", "увеличить", "достичь"],
    "policy": ["политика", "кодекс", "регламент", "стандарт", "положение", "стратегия"],
    "process": ["порядок", "процедура", "процесс", "рассмотрение", "обработка", "контроль"],
    "role": ["комитет", "ответственный", "директор", "служба", "управление"],
    "report": ["отчет", "отчёт", "раскрытие", "годовой", "нефинансов", "интегрированный"],
}

NEGATIVE_NAV_MARKERS = [
    "cookie", "cookies", "javascript",
    "включите javascript", "личный кабинет", "поиск по сайту",
    "карта сайта", "версия для слабовидящих", "все права защищены",
    "поделиться", "наверх", "меню", "навигация", "главная страница",
    "читать далее",
]

SOURCE_PRIORITY_MARKERS = [
    ".pdf", "отчет", "отчёт", "esg", "устойчив", "сертификат", "iso",
    "кодекс", "политик", "положение", "раскрытие", "документ", "закуп",
    "горячая линия", "антикорруп", "качество", "охрана труда",
]

# =====================================================================
# Семантический конфиг. В локальной версии на VM эти имена уже
# определены — тогда значения берутся оттуда, здесь только дефолты.
# =====================================================================

RAG_ENABLE_SEMANTIC = os.environ.get("RAG_ENABLE_SEMANTIC", "1") == "1"
RAG_SEMANTIC_WEIGHT = float(os.environ.get("RAG_SEMANTIC_WEIGHT", "9.0")) # изначально 19

# RAG_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "intfloat/multilingual-e5-base")
RAG_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")
RAG_EMBED_BATCH = int(os.environ.get("RAG_EMBED_BATCH", "16"))

_META_LINE_RE = re.compile(
    r"^(url|date|title|domain|inn|created|lastmodified|компания)\s*:", re.I
)


def split_meta_body(text):
    """Разделить текст чанка на мета-строки (URL:/DATE:/TITLE: ...) и тело."""
    meta, body = [], []
    for ln in str(text or "").splitlines():
        (meta if _META_LINE_RE.match(ln.strip()) else body).append(ln)
    return "\n".join(meta), "\n".join(body)


def chunk_embed_text(chunk):
    """
    Текст для эмбеддинга: тело чанка БЕЗ повторяющегося мета-header'а.
    У e5-base лимит 512 токенов — header съедает лимит, и контент
    чанка обрезается токенизатором.
    """
    _, body = split_meta_body(chunk.get("text", ""))
    body = body.strip()
    return body if body else str(chunk.get("text", ""))


def _unique_with_multiplicity(items):
    """Уникальные строки с кратностью: дубликаты в списках маркеров суммируются."""
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return counts


# =====================================================================
# Векторизованный лексический скорер (оптимизация п.4)
#
# Раньше chunk_relevance_score() вызывался поштучно для каждой пары
# (переменная × чанк): 15 переменных × ~5000 чанков = ~75k вызовов,
# с повторной нормализацией текста и пересборкой терминов запроса.
#
# Здесь всё, что не зависит от переменной (нормализованные тексты,
# presence терминов, маркеры evidence/priority, regex-признаки,
# константы чанка), считается ОДИН раз на компанию; score переменной —
# несколько numpy-операций над векторами длины n_chunks.
# Семантика scoring'а сохранена 1:1, включая кратность дублей маркеров
# вида 'отчет'/'отчёт'. Единственное сознательное отличие: проверка
# фразы/термина в title/url/file идёт по объединённой строке blob
# вместо трёх отдельных проверок (теоретически возможен ложный матч
# на стыке строк — на практике несущественно).
# =====================================================================
class LexicalScorer:
    def __init__(self, chunks):
        t0 = time.perf_counter()
        self.chunks = chunks
        n = self.n = len(chunks)

        self.lows = [_norm_text_for_score(ch.get("text", "")) for ch in chunks]
        # self.lows = [
        #     stem_text(
        #         _norm_text_for_score(
        #             ch.get("text","")
        #         )
        #     )
        #     for ch in chunks
        # ]
        self.blobs = [
            " ".join([
                _norm_text_for_score(ch.get("title", "")),
                _norm_text_for_score(ch.get("url", "")),
                _norm_text_for_score(ch.get("file_path", "")),
            ])
            for ch in chunks
        ]
        self.head_blobs = [
            " ".join((low[:4000], blob)) for low, blob in zip(self.lows, self.blobs)
        ]

        # --- термины запросов всех переменных: presence-матрицы ---
        self.var_terms = {var: tokenize_query_terms(qs) for var, qs in ESG_QUERIES.items()}
        vocab = sorted(set().union(*self.var_terms.values())) if self.var_terms else []
        self.vocab = vocab
        self.vocab_index = {t: i for i, t in enumerate(vocab)}
        m = len(vocab)

        self.body_pres = np.zeros((n, m), dtype=bool)
        self.blob_pres = np.zeros((n, m), dtype=bool)
        for i in range(n):
            low, blob = self.lows[i], self.blobs[i]
            self.body_pres[i] = [t in low for t in vocab]
            self.blob_pres[i] = [t in blob for t in vocab]

        # --- evidence-маркеры: сабстринг-проверки, дубликаты суммируются ---
        ev_counts = _unique_with_multiplicity(
            _norm_text_for_score(x)
            for marks in EVIDENCE_MARKERS.values() for x in marks
        )
        self.evidence_score = np.zeros(n)
        for marker, mult in ev_counts.items():
            if marker:
                # self.evidence_score += 0.9 * mult * np.fromiter(
                self.evidence_score += 0.45 * mult * np.fromiter(
                    # (marker in low for low in self.lows), dtype=np.float64, count=n
                    (contains_term(low, marker) for low in self.lows), dtype=np.float64, count=n
                )

        # --- priority-маркеры: по source_blob (тело[:4000] + title + url + file) ---
        pr_counts = _unique_with_multiplicity(
            _norm_text_for_score(x) for x in SOURCE_PRIORITY_MARKERS
        )
        self.priority_score = np.zeros(n)
        for marker, mult in pr_counts.items():
            if marker:
                # self.priority_score += 1.0 * mult * np.fromiter(
                self.priority_score += 0.5 * mult * np.fromiter(
                    # (marker in hb for hb in self.head_blobs), dtype=np.float64, count=n
                    (contains_term(hb, marker) for hb in self.head_blobs), dtype=np.float64, count=n
                )
        print(
            f"[CHECK WEIGHTS] evidence_weight=0.45 priority_weight=0.5",
            flush=True
        )

        # --- regex-признаки (год 20xx, число+единица измерения) ---
        _year_re = re.compile(r"\b20\d{2}\b")
        _num_re = re.compile(r"\d+[,.]?\d*\s*(%|тонн|т|м3|м²|квт|квт·ч|руб|млн|млрд)")
        self.regex_score = np.array([
            (1.5 if _year_re.search(low) else 0.0) + (1.8 if _num_re.search(low) else 0.0)
            for low in self.lows
        ])

        # --- константы чанка: бонусы за мету, штрафы за мусор ---
        const = np.zeros(n)
        for i, ch in enumerate(chunks):
            low = self.lows[i]
            text_len = len(str(ch.get("text", "")))
            c = 0.0
            if "url:" in low or ch.get("url"):
                c += 1.2
            if "date:" in low or "created:" in low or "lastmodified:" in low or ch.get("date"):
                c += 0.8
            if "title:" in low or ch.get("title"):
                c += 0.5
            if text_len < 220:
                c -= 2.0
            if sum(1 for x in NEGATIVE_NAV_MARKERS if x in low) >= 2 and text_len < 1500:
                c -= 3.0
            if low.count("http") > 10 and text_len < 1600:
                c -= 1.5
            const[i] = c
        self.const = const

        self._phrase_cache = {}
        self.build_time = time.perf_counter() - t0

    def score(self, var, stage_queries=None, *, include_negative=True):
        """Вектор лексических score для переменной, np.array shape (n_chunks,).

        include_negative=False — считает score БЕЗ stage-2 negative penalty
        (нужно только диагностическому режиму RAG_DIAGNOSTIC для A/B-разбивки
        positive_score / negative_penalty). Прод-поведение не меняется.
        """
        queries = stage_queries if stage_queries is not None else ESG_QUERIES.get(var, [])
        terms = tokenize_query_terms(queries)

        ids = [self.vocab_index[t] for t in terms if t in self.vocab_index]
        if ids:
            id_arr = np.asarray(ids, dtype=int)
            hits_body = self.body_pres[:, id_arr].sum(axis=1).astype(np.float64)
            hits_blob = self.blob_pres[:, id_arr].sum(axis=1).astype(np.float64)
        else:
            hits_body = np.zeros(self.n)
            hits_blob = np.zeros(self.n)

        k = max(1, len(terms))
        scores = self.const + self.evidence_score + self.priority_score + self.regex_score
        scores = scores + hits_body * (1.4 + 8.0 / k) + 1.1 * hits_blob

        for ph in queries:
            p = _norm_text_for_score(ph)
            if not p:
                continue
            if p not in self._phrase_cache:
                self._phrase_cache[p] = (
                    np.fromiter((p in low for low in self.lows), dtype=np.float64, count=self.n),
                    np.fromiter((p in b for b in self.blobs), dtype=np.float64, count=self.n),
                )
            pb, pbl = self._phrase_cache[p]
            scores = scores + 5.0 * pb + 3.0 * pbl

        # --- штраф за совпадение с темой, с которой var обычно путают ---
        # for ph in NEGATIVE_QUERIES.get(var, []):
        #     p = _norm_text_for_score(ph)
        #     if not p:
        #         continue
        #     if p not in self._phrase_cache:
        #         self._phrase_cache[p] = (
        #             np.fromiter((p in low for low in self.lows), dtype=np.float64, count=self.n),
        #             np.fromiter((p in b for b in self.blobs), dtype=np.float64, count=self.n),
        #         )
        #     pb, pbl = self._phrase_cache[p]
        #     scores = scores - 2.5 * pb - 1.5 * pbl

                # ---------------------------------------------------------------------
        # МЯГКИЙ ШТРАФ ЗА БЛИЗКИЕ, НО НЕЦЕЛЕВЫЕ ТЕМЫ
        #
        # Важно:
        # - применяется только к Stage 2;
        # - штрафуется категория целиком, а не каждое слово отдельно;
        # - несколько совпадений внутри одной категории не увеличивают штраф;
        # - это только reranking, а не hard filter.
        # ---------------------------------------------------------------------

        if include_negative and var in STAGE2_VARIABLES:
            negative_categories = NEGATIVE_CATEGORIES_BY_VAR.get(var, [])

            for category in negative_categories:
                category_hit_body = np.zeros(self.n, dtype=np.float64)
                category_hit_blob = np.zeros(self.n, dtype=np.float64)

                for ph in UNIVERSAL_NEGATIVE_TERMS.get(category, []):
                    p = _norm_text_for_score(ph)
                    if not p:
                        continue

                    if p not in self._phrase_cache:
                        self._phrase_cache[p] = (
                            np.fromiter(
                                (p in low for low in self.lows),
                                dtype=np.float64,
                                count=self.n
                            ),
                            np.fromiter(
                                (p in b for b in self.blobs),
                                dtype=np.float64,
                                count=self.n
                            ),
                        )

                    pb, pbl = self._phrase_cache[p]

                    category_hit_body = np.maximum(category_hit_body, pb)
                    category_hit_blob = np.maximum(category_hit_blob, pbl)

                # Один штраф за одну смысловую категорию,
                # независимо от количества найденных терминов.
                scores = (
                    scores
                    - 1.2 * category_hit_body
                    - 0.7 * category_hit_blob
                )

        return scores


# =====================================================================
# Эмбеддинги: модель-синглтон на процесс + батчевое кодирование
# =====================================================================
_EMBED_MODEL = None
_EMBED_QUERY_CACHE = {}


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(int(os.environ.get("RAG_TORCH_THREADS") or (os.cpu_count() or 4)))
        _EMBED_MODEL = SentenceTransformer(RAG_EMBED_MODEL)
        print(f"[semantic] Модель загружена: {RAG_EMBED_MODEL} "
              f"(device={_EMBED_MODEL.device}, threads={torch.get_num_threads()})")
    return _EMBED_MODEL


def compute_chunk_embeddings(chunks, batch_size=None):
    """Один батчевой encode на всю компанию вместо поштучных вызовов."""
    model = _get_embed_model()
    texts = ["passage: " + chunk_embed_text(ch) for ch in chunks]
    t0 = time.perf_counter()
    emb = model.encode(
        texts,
        batch_size=batch_size or RAG_EMBED_BATCH,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dt = time.perf_counter() - t0
    print(f"[semantic] Эмбеддингов посчитано: {len(chunks)} за {dt:.1f} с "
          f"({len(chunks) / max(dt, 1e-9):.1f} чанк/с)")
    return emb

def _normalize_doc_text_for_dedupe(text):
    """
    Нормализация текста для поиска технических дублей.

    Убираем различия форматирования, пробелы и переводы строк,
    но не используем агрессивную очистку, которая могла бы
    превратить документы разных периодов в одинаковые.
    """
    text = str(text or "")

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Схлопываем пробельные последовательности.
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def _doc_content_hash(text):
    """
    Стабильный hash нормализованного полного текста документа.
    """
    normalized = _normalize_doc_text_for_dedupe(text)

    return hashlib.sha256(
        normalized.encode("utf-8", errors="ignore")
    ).hexdigest()


def _doc_length_ratio(text_a, text_b):
    """
    Отношение длины меньшего текста к большему.
    1.0 = одинаковая длина.
    """
    a = len(_normalize_doc_text_for_dedupe(text_a))
    b = len(_normalize_doc_text_for_dedupe(text_b))

    if a == 0 or b == 0:
        return 0.0

    return min(a, b) / max(a, b)


def _doc_edge_match(text_a, text_b, edge_chars=RAG_DOC_DUP_EDGE_CHARS):
    """
    Проверяет совпадение начала и конца документов.

    Возвращает среднюю долю совпадения.
    """
    a = _normalize_doc_text_for_dedupe(text_a)
    b = _normalize_doc_text_for_dedupe(text_b)

    if not a or not b:
        return 0.0

    n = min(edge_chars, len(a), len(b))

    if n <= 0:
        return 0.0

    start_match = 1.0 if a[:n] == b[:n] else 0.0
    end_match = 1.0 if a[-n:] == b[-n:] else 0.0

    return (start_match + end_match) / 2.0


def dedupe_similar_docs(
    chunks,
    docs,
    threshold=RAG_DOC_DEDUPE_THRESHOLD,
):
    """
    Исключает технические дубли документов из текущего RAG retrieval.

    ВАЖНО:
    - исходные документы на диске НЕ удаляются;
    - docs НЕ изменяются;
    - chunks удаляются только из текущей RAG-сессии;
    - документы разных периодов не объединяются, если период
      явно различается;
    - отсутствие даты НЕ является основанием для удаления
      похожего документа.

    Логика:

    1. Полный дубль:
       одинаковый hash нормализованного полного текста.

    2. Практически полный дубль:
       очень высокая document-level embedding similarity +
       близкая длина текста +
       совпадение начала/конца текста.

    3. Если даты явно конфликтуют:
       документы НЕ считаются дублями.

    Для каждой группы дублей остаётся один представитель.
    """

    if not chunks or not docs:
        return chunks

    # ------------------------------------------------------------
    # 1. Метаданные документов
    # ------------------------------------------------------------

    doc_dates = {}
    doc_lens = {}
    doc_hashes = {}
    doc_normalized_texts = {}

    for i, d in enumerate(docs):
        text = str(d.get("text", "") or "")

        normalized = _normalize_doc_text_for_dedupe(text)

        doc_normalized_texts[i] = normalized
        doc_lens[i] = len(normalized)

        doc_hashes[i] = hashlib.sha256(
            normalized.encode("utf-8", errors="ignore")
        ).hexdigest()

        doc_dates[i] = (
            d.get("date")
            or d.get("created")
            or d.get("lastmodified")
            or d.get("last_modified")
            or ""
        )

    # ------------------------------------------------------------
    # 2. Индекс чанков по документам
    # ------------------------------------------------------------

    by_doc = {}

    for ch in chunks:
        doc_i = ch.get("doc_i")

        if doc_i is None:
            continue

        by_doc.setdefault(doc_i, []).append(ch)

    candidate_ids = sorted(
        doc_i
        for doc_i in by_doc
        if 0 <= doc_i < len(docs)
        and doc_lens.get(doc_i, 0) > 0
    )

    if not candidate_ids:
        return chunks

    # ------------------------------------------------------------
    # 3. Document-level vectors
    #
    # Используем уже рассчитанные embeddings чанков.
    # Это НЕ требует повторного вызова embedding model.
    # ------------------------------------------------------------

    by_doc_vecs = {}

    for doc_i in candidate_ids:
        vecs = [
            ch["embedding"]
            for ch in by_doc.get(doc_i, [])
            if ch.get("embedding") is not None
        ]

        if not vecs:
            continue

        v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(v)

        if norm > 0:
            by_doc_vecs[doc_i] = v / norm

    # ------------------------------------------------------------
    # 4. Группы точных дублей по hash
    # ------------------------------------------------------------

    hash_groups = {}

    for doc_i in candidate_ids:
        h = doc_hashes.get(doc_i)

        if h:
            hash_groups.setdefault(h, []).append(doc_i)

    duplicate_groups = []
    assigned = set()

    # ------------------------------------------------------------
    # 5. Точные дубли
    # ------------------------------------------------------------

    for h, group in hash_groups.items():

        if len(group) < 2:
            continue

        # Проверяем даты.
        safe_group = []

        for doc_i in group:
            conflict = False

            for other_i in group:
                if doc_i == other_i:
                    continue

                if _dates_conflict(
                    doc_dates.get(doc_i),
                    doc_dates.get(other_i),
                    file_path_a=docs[doc_i].get("file_path", ""),
                    file_path_b=docs[other_i].get("file_path", ""),
                ):
                    conflict = True
                    break

            if not conflict:
                safe_group.append(doc_i)

        # Если одинаковый текст, но явно разные даты,
        # не считаем документы дублями.
        if len(safe_group) >= 2:
            duplicate_groups.append({
                "type": "exact",
                "docs": safe_group,
            })

            assigned.update(safe_group)

    # ------------------------------------------------------------
    # 6. Near-duplicates
    #
    # Только очень высокая similarity.
    # Уже найденные exact duplicates здесь не рассматриваем.
    # ------------------------------------------------------------

    remaining_ids = [
        doc_i
        for doc_i in candidate_ids
        if doc_i not in assigned
        and doc_i in by_doc_vecs
    ]

    near_duplicate_pairs = []

    for pos, di in enumerate(remaining_ids):

        for dj in remaining_ids[pos + 1:]:

            # ----------------------------------------------------
            # 6.1. Защита от разных дат/периодов
            # ----------------------------------------------------

            if _dates_conflict(
                doc_dates.get(di),
                doc_dates.get(dj),
                file_path_a=docs[di].get("file_path", ""),
                file_path_b=docs[dj].get("file_path", ""),
            ):
                continue

            # ----------------------------------------------------
            # 6.2. Сходство embeddings
            # ----------------------------------------------------

            sim = float(
                by_doc_vecs[di] @ by_doc_vecs[dj]
            )

            if sim < RAG_DOC_NEAR_DUP_THRESHOLD:
                continue

            # ----------------------------------------------------
            # 6.3. Длина документов
            # ----------------------------------------------------

            length_ratio = (
                min(doc_lens[di], doc_lens[dj])
                / max(doc_lens[di], doc_lens[dj])
            )

            if length_ratio < RAG_DOC_NEAR_DUP_LENGTH_RATIO:
                continue

            # ----------------------------------------------------
            # 6.4. Начало и конец документа
            # ----------------------------------------------------

            edge_match = _doc_edge_match(
                doc_normalized_texts[di],
                doc_normalized_texts[dj],
            )

            # Для near-duplicate требуется совпадение
            # либо начала, либо конца документа.
            #
            # Для полностью одинаковых документов этот показатель
            # равен 1.0.
            if edge_match < 0.5:
                continue

            near_duplicate_pairs.append(
                (di, dj, sim, length_ratio, edge_match)
            )

    # ------------------------------------------------------------
    # 7. Формируем группы near-duplicates
    # ------------------------------------------------------------

    for di, dj, sim, length_ratio, edge_match in near_duplicate_pairs:

        # Если документ уже вошёл в другую группу,
        # не создаём новую конкурирующую группу.
        existing_group = None

        for group in duplicate_groups:
            if di in group["docs"] or dj in group["docs"]:
                existing_group = group
                break

        if existing_group is not None:
            if di not in existing_group["docs"]:
                existing_group["docs"].append(di)
            if dj not in existing_group["docs"]:
                existing_group["docs"].append(dj)

        else:
            duplicate_groups.append({
                "type": "near",
                "docs": [di, dj],
            })

    # ------------------------------------------------------------
    # 8. Выбираем представителя каждой группы
    #
    # Оставляем самый длинный документ.
    # При одинаковой длине используется меньший doc_i,
    # поэтому результат стабилен.
    # ------------------------------------------------------------

    dropped = set()
    merge_log = []

    for group in duplicate_groups:

        group_docs = [
            i for i in group["docs"]
            if i not in dropped
        ]

        if len(group_docs) < 2:
            continue

        winner = sorted(
            group_docs,
            key=lambda i: (-doc_lens.get(i, 0), i)
        )[0]

        losers = [
            i for i in group_docs
            if i != winner
        ]

        dropped.update(losers)

        def _doc_label(doc_i):
            d = docs[doc_i]

            fp = str(
                d.get("file_path", "") or ""
            ).split("/")[-1]

            title = str(
                d.get("title", "") or ""
            )

            url = str(
                d.get("url", "") or ""
            )

            parts = [f"doc_{doc_i}"]

            if fp:
                parts.append(fp)

            if title:
                parts.append(f"title={title[:100]}")

            if url:
                parts.append(url[:150])

            return " | ".join(parts)

        merge_log.append(
            f"      type={group['type']} | "
            f"group={[f'doc_{i}' for i in group_docs]} | "
            f"keep=[doc_{winner}] {_doc_label(winner)} | "
            f"drop={[f'doc_{i}' for i in losers]}"
        )

    # ------------------------------------------------------------
    # 9. Логирование
    # ------------------------------------------------------------

    exact_groups = sum(
        1 for g in duplicate_groups
        if g["type"] == "exact"
    )

    near_groups = sum(
        1 for g in duplicate_groups
        if g["type"] == "near"
    )

    if duplicate_groups:
        print(
            f"    [dedupe_docs] "
            f"групп дублей: {len(duplicate_groups)} | "
            f"exact={exact_groups} | "
            f"near={near_groups} | "
            f"исключено документов: {len(dropped)}"
        )

        for line in merge_log:
            print(line)

    else:
        print(
            "    [dedupe_docs] "
            "дубликатов не найдено"
        )

    # ------------------------------------------------------------
    # 10. ВАЖНО:
    #     ничего не удаляем из docs или с диска.
    #
    #     Просто не передаём чанки этих документов дальше.
    # ------------------------------------------------------------

    if not dropped:
        return chunks

    filtered_chunks = [
        ch
        for ch in chunks
        if ch.get("doc_i") not in dropped
    ]

    return filtered_chunks

def encode_stage_query(var, stage_queries=None):
    """Эмбеддинг запроса переменной — с кэшем на процесс (15 переменных = 15 encode, не больше)."""
    key = (var, json.dumps(stage_queries or ESG_QUERIES.get(var, []),
                           ensure_ascii=False, sort_keys=True))
    if key not in _EMBED_QUERY_CACHE:
        model = _get_embed_model()
        queries = stage_queries or ESG_QUERIES.get(var, [])
        _EMBED_QUERY_CACHE[key] = model.encode(
            ["query: " + "; ".join(queries)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
    return _EMBED_QUERY_CACHE[key]


# =====================================================================
# In-memory кэш одной компании (оптимизация п.3)
#
# Создаётся в начале обработки компании, выбрасывается после —
# на диск ничего не пишется, между компаниями состояние не течёт.
# Кэширует: docs -> chunks -> лексический индекс -> score-векторы
# переменных (лексика + семантика).
# =====================================================================
class CompanyRAGSession:
    def __init__(self, company_id, *, sources_dir=SOURCES_DIR,
                 max_chars_per_chunk=1600,
                 enable_semantic=None, semantic_weight=None):
        self.company_id = company_id
        self.sources_dir = sources_dir
        self.max_chars_per_chunk = max_chars_per_chunk
        self.enable_semantic = RAG_ENABLE_SEMANTIC if enable_semantic is None else bool(enable_semantic)
        self.semantic_weight = RAG_SEMANTIC_WEIGHT if semantic_weight is None else float(semantic_weight)
        self.timings = {}
        self._docs = None
        self._chunks = None
        self._scorer = None
        self._embeddings = None
        self._score_by_var = {}

    def _tick(self, name, t0):
        self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - t0)

    def report_timings(self):
        return " | ".join(f"{k}={v:.1f}s" for k, v in sorted(self.timings.items()))

    # ---- этап 1: чтение документов ----
    def get_docs(self, company_sources):
        if self._docs is None:
            t0 = time.perf_counter()
            self._docs = read_company_docs(company_sources)
            self._tick("1_read_docs", t0)
        return self._docs

    # ---- этап 2: чанки (+ эмбеддинги батчем, если семантика включена) ----
    def get_chunks(self, company_sources=None):
        if self._chunks is not None:
            return self._chunks
        docs = self.get_docs(company_sources) if company_sources is not None else self._docs
        t0 = time.perf_counter()
        chunks = build_all_chunks(docs, max_chars_per_chunk=self.max_chars_per_chunk)
        self._tick("2_build_chunks", t0)
        if self.enable_semantic:
            t0 = time.perf_counter()
            self._embeddings = compute_chunk_embeddings(chunks)
            for ch, v in zip(chunks, self._embeddings):
                ch["embedding"] = v
            self._tick("3_embed", t0)

            t0 = time.perf_counter()
            chunks = dedupe_similar_docs(chunks, docs, threshold=RAG_DOC_DEDUPE_THRESHOLD)
            self._embeddings = np.array([ch["embedding"] for ch in chunks])
            self._tick("3b_dedupe_docs", t0)
        self._chunks = chunks
        return self._chunks

    # ---- этап 3: лексический индекс (один на компанию) ----
    def get_scorer(self):
        if self._scorer is None:
            self._scorer = LexicalScorer(self.get_chunks())
            self.timings["4_lex_index"] = self._scorer.build_time
        return self._scorer

    # ---- этап 4: score-вектор переменной (лексика + семантика), кэш на сессию ----
    def get_scores(self, var, stage_queries=None):
        if var in self._score_by_var:
            return self._score_by_var[var]
        t0 = time.perf_counter()
        scores = self.get_scorer().score(var, stage_queries).copy()
        if self.enable_semantic and self._embeddings is not None:
            q = encode_stage_query(var, stage_queries)
            sim = self._embeddings @ q  # обе нормированы -> косинусная близость
            scores += self.semantic_weight * sim
        self._score_by_var[var] = scores
        self._tick("5_score_vars", t0)
        return scores

# =====================================================================
# ДИАГНОСТИКА СОСТАВЛЯЮЩИХ RETRIEVAL SCORE
# БЕЗ LLM / БЕЗ ЗАПИСИ НА VM / БЕЗ ИЗМЕНЕНИЯ PRODUCTION CONFIG
# =====================================================================

def diagnose_retrieval_components(
    session,
    var,
    *,
    stage_queries=None,
    top_k=50,
    show_all=False,
):
    """
    Подробно разбирает текущий retrieval score для одной переменной.

    Использует ИМЕННО текущую production-логику:
        lexical score
        + semantic_weight * cosine_similarity

    Для lexical score дополнительно показывает:
        - body term contribution
        - metadata term contribution
        - phrase contribution в body
        - phrase contribution в metadata
        - coverage
        - evidence markers
        - source priority
        - year bonus
        - numeric bonus
        - URL/date/title bonuses
        - negative penalty

    ВАЖНО:
    - LLM не вызывается;
    - файлы не изменяются;
    - глобальные настройки не меняются;
    - scorer/session не модифицируются;
    - используется уже существующая embedding-модель и
      уже рассчитанные embeddings текущей session.

    Возвращает DataFrame, чтобы при необходимости его можно было
    дополнительно анализировать программно.
    """

    scorer = session.get_scorer()
    chunks = session.get_chunks()

    # ------------------------------------------------------------
    # 1. Production lexical score
    # ------------------------------------------------------------

    lexical_full = scorer.score(
        var,
        stage_queries,
        include_negative=True,
    )

    # То же самое, но без текущего negative penalty.
    lexical_positive = scorer.score(
        var,
        stage_queries,
        include_negative=False,
    )

    negative_penalty = lexical_full - lexical_positive

    # ------------------------------------------------------------
    # 2. Production semantic contribution
    # ------------------------------------------------------------

    semantic_cosine = np.zeros(len(chunks), dtype=float)
    semantic_weighted = np.zeros(len(chunks), dtype=float)

    if (
        getattr(session, "enable_semantic", False)
        and getattr(session, "_embeddings", None) is not None
    ):
        q = encode_stage_query(var, stage_queries)
        semantic_cosine = session._embeddings @ q
        semantic_weighted = session.semantic_weight * semantic_cosine

    # Production final score.
    final_scores = lexical_full + semantic_weighted

    # ------------------------------------------------------------
    # 3. Query terms / phrases
    # ------------------------------------------------------------

    queries = (
        stage_queries
        if stage_queries is not None
        else ESG_QUERIES.get(var, [])
    )

    q_terms = tokenize_query_terms(queries)
    n_terms = max(1, len(q_terms))

    # ------------------------------------------------------------
    # 4. Ranking
    # ------------------------------------------------------------

    order = np.argsort(-final_scores, kind="stable")

    rows = []

    for rank_zero, idx in enumerate(order[:top_k]):
        idx = int(idx)
        ch = chunks[idx]

        text = str(ch.get("text", "") or "")
        low = _norm_text_for_score(text)

        title = str(ch.get("title", "") or "")
        url = str(ch.get("url", "") or "")
        file_path = str(ch.get("file_path", "") or "")

        title_low = _norm_text_for_score(title)
        url_low = _norm_text_for_score(url)
        file_low = _norm_text_for_score(file_path)

        # --------------------------------------------------------
        # Body terms
        # --------------------------------------------------------

        body_term_hits = sum(
            1 for t in q_terms
            if t in low
        )

        body_term_score = (
            body_term_hits * (1.4 + 8.0 / n_terms)
        )

        # --------------------------------------------------------
        # Metadata terms
        # --------------------------------------------------------

        metadata_term_hits = sum(
            1
            for t in q_terms
            if (
                t in title_low
                or t in url_low
                or t in file_low
            )
        )

        metadata_term_score = metadata_term_hits * 1.1

        # --------------------------------------------------------
        # Phrase matches
        # --------------------------------------------------------

        phrase_body_hits = []
        phrase_metadata_hits = []

        phrase_body_score = 0.0
        phrase_metadata_score = 0.0

        for phrase in queries:
            p = _norm_text_for_score(phrase)

            if not p:
                continue

            if p in low:
                phrase_body_hits.append(phrase)
                phrase_body_score += 5.0

            if (
                p in title_low
                or p in url_low
                or p in file_low
            ):
                phrase_metadata_hits.append(phrase)
                phrase_metadata_score += 3.0

        # --------------------------------------------------------
        # Coverage
        # --------------------------------------------------------

        coverage = body_term_hits / n_terms
        coverage_score = 8.0 * coverage

        # --------------------------------------------------------
        # Evidence markers
        # --------------------------------------------------------

        evidence_hits = []

        for category, markers in EVIDENCE_MARKERS.items():
            for marker in markers:
                marker_norm = _norm_text_for_score(marker)

                if marker_norm and marker_norm in low:
                    evidence_hits.append(
                        f"{category}:{marker}"
                    )

        evidence_score = 0.9 * len(evidence_hits)

        # --------------------------------------------------------
        # Source priority
        # --------------------------------------------------------

        source_blob = " ".join([
            low[:4000],
            title_low,
            url_low,
            file_low,
        ])

        source_priority_hits = []

        for marker in SOURCE_PRIORITY_MARKERS:
            marker_norm = _norm_text_for_score(marker)

            if marker_norm and marker_norm in source_blob:
                source_priority_hits.append(marker)

        source_priority_score = 1.0 * len(source_priority_hits)

        # --------------------------------------------------------
        # Regex bonuses
        # --------------------------------------------------------

        has_year = bool(
            re.search(r"\b20\d{2}\b", low)
        )

        has_numeric_evidence = bool(
            re.search(
                r"\d+[,.]?\d*\s*"
                r"(%|тонн|т|м3|м²|квт|квт·ч|руб|млн|млрд)",
                low
            )
        )

        year_score = 1.5 if has_year else 0.0
        numeric_score = 1.8 if has_numeric_evidence else 0.0

        # --------------------------------------------------------
        # Technical metadata bonuses
        # --------------------------------------------------------

        url_bonus = (
            1.2
            if ("url:" in low or ch.get("url"))
            else 0.0
        )

        date_bonus = (
            0.8
            if (
                "date:" in low
                or "created:" in low
                or "lastmodified:" in low
                or ch.get("date")
            )
            else 0.0
        )

        title_bonus = (
            0.5
            if ("title:" in low or ch.get("title"))
            else 0.0
        )

        short_text_penalty = (
            -2.0
            if len(text) < 220
            else 0.0
        )

        nav_marker_penalty = (
            -3.0
            if (
                sum(
                    1
                    for x in NEGATIVE_NAV_MARKERS
                    if x in low
                ) >= 2
                and len(text) < 1500
            )
            else 0.0
        )

        many_http_penalty = (
            -1.5
            if (
                low.count("http") > 10
                and len(text) < 1600
            )
            else 0.0
        )

        technical_penalty = (
            short_text_penalty
            + nav_marker_penalty
            + many_http_penalty
        )

        # --------------------------------------------------------
        # Independent reconstruction of lexical score
        #
        # Нужна как контроль того, что диагностика действительно
        # соответствует production scorer.
        # --------------------------------------------------------

        lexical_reconstructed = (
            body_term_score
            + metadata_term_score
            + phrase_body_score
            + phrase_metadata_score
            + coverage_score
            + evidence_score
            + source_priority_score
            + year_score
            + numeric_score
            + url_bonus
            + date_bonus
            + title_bonus
            + technical_penalty
            + float(negative_penalty[idx])
        )

        # --------------------------------------------------------
        # Negative categories
        # --------------------------------------------------------

        negative_categories_hit = []

        if var in STAGE2_VARIABLES:
            negative_categories = (
                NEGATIVE_CATEGORIES_BY_VAR.get(var, [])
            )

            for category in negative_categories:

                category_hit = False

                for phrase in UNIVERSAL_NEGATIVE_TERMS.get(
                    category, []
                ):
                    p = _norm_text_for_score(phrase)

                    if not p:
                        continue

                    if p in low or p in " ".join(
                        [title_low, url_low, file_low]
                    ):
                        category_hit = True
                        break

                if category_hit:
                    negative_categories_hit.append(category)

        # --------------------------------------------------------
        # Document id
        # --------------------------------------------------------

        document_id = (
            ch.get("document_id")
            or ch.get("doc_id")
            or ch.get("document")
            or ""
        )

        chunk_id = (
            ch.get("chunk_id")
            or ch.get("id")
            or ""
        )

        # --------------------------------------------------------
        # Rank without negative
        # --------------------------------------------------------

        # Будет вычислено ниже один раз для всех строк.
        # Пока сохраняем исходные значения.
        rows.append({
            "rank": rank_zero + 1,
            "index": idx,

            "document_id": document_id,
            "chunk_id": chunk_id,

            "final_score": float(final_scores[idx]),

            "lexical_score": float(lexical_full[idx]),
            "lexical_positive": float(lexical_positive[idx]),
            "negative_penalty": float(negative_penalty[idx]),

            "semantic_cosine": float(semantic_cosine[idx]),
            "semantic_weighted": float(semantic_weighted[idx]),

            "body_term_hits": int(body_term_hits),
            "body_term_score": float(body_term_score),

            "metadata_term_hits": int(metadata_term_hits),
            "metadata_term_score": float(metadata_term_score),

            "phrase_body_hits": len(phrase_body_hits),
            "phrase_body_score": float(phrase_body_score),

            "phrase_metadata_hits": len(phrase_metadata_hits),
            "phrase_metadata_score": float(phrase_metadata_score),

            "coverage": float(coverage),
            "coverage_score": float(coverage_score),

            "evidence_hits": len(evidence_hits),
            "evidence_score": float(evidence_score),

            "source_priority_hits": len(source_priority_hits),
            "source_priority_score": float(source_priority_score),

            "year_score": float(year_score),
            "numeric_score": float(numeric_score),

            "url_bonus": float(url_bonus),
            "date_bonus": float(date_bonus),
            "title_bonus": float(title_bonus),

            "technical_penalty": float(technical_penalty),

            "negative_categories": ", ".join(
                negative_categories_hit
            ),

            "lexical_reconstructed": float(
                lexical_reconstructed
            ),

            "lexical_reconstruction_error": float(
                lexical_reconstructed - lexical_full[idx]
            ),

            "title": title,
            "url": url,
            "file_path": file_path,

            "text_len": len(text),
            "text_preview": re.sub(
                r"\s+",
                " ",
                text
            )[:500],
        })

    # ------------------------------------------------------------
    # 5. Rank without negative
    # ------------------------------------------------------------

    order_without_negative = np.argsort(
        -(
            lexical_positive
            + semantic_weighted
        ),
        kind="stable",
    )

    rank_without_negative = np.empty(
        len(chunks),
        dtype=int,
    )

    rank_without_negative[
        order_without_negative
    ] = np.arange(
        1,
        len(chunks) + 1,
    )

    for row in rows:
        row["rank_without_negative"] = int(
            rank_without_negative[row["index"]]
        )

        row["rank_shift_due_to_negative"] = (
            row["rank_without_negative"]
            - row["rank"]
        )

    # ------------------------------------------------------------
    # 6. DataFrame
    # ------------------------------------------------------------

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 7. Человекочитаемый вывод
    # ------------------------------------------------------------

    print()
    print("=" * 120)
    print(
        f"RETRIEVAL SCORE DIAGNOSTIC | "
        f"variable={var} | "
        f"chunks={len(chunks)} | "
        f"top_k={top_k}"
    )
    print("=" * 120)

    print(
        f"semantic_enabled={session.enable_semantic} | "
        f"semantic_weight={session.semantic_weight}"
    )

    print()

    print(
        "Колонки:"
    )
    print(
        "final = lexical + semantic_weighted"
    )
    print(
        "lexical = positive lexical + negative penalty"
    )

    print()

    # ------------------------------------------------------------
    # 8. Основная таблица
    # ------------------------------------------------------------

    display_cols = [
        "rank",
        "document_id",
        "chunk_id",
        "final_score",
        "lexical_score",
        "semantic_weighted",
        "semantic_cosine",
        "lexical_positive",
        "negative_penalty",
        "body_term_score",
        "metadata_term_score",
        "phrase_body_score",
        "phrase_metadata_score",
        "coverage_score",
        "evidence_score",
        "source_priority_score",
        "year_score",
        "numeric_score",
        "url_bonus",
        "date_bonus",
        "title_bonus",
        "technical_penalty",
        "rank_without_negative",
        "rank_shift_due_to_negative",
    ]

    if show_all:
        print(
            df[display_cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )
    else:
        print(
            df[display_cols].head(top_k).to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    # ------------------------------------------------------------
    # 9. Контроль совпадения реконструированного lexical score
    # ------------------------------------------------------------

    max_error = float(
        df["lexical_reconstruction_error"]
        .abs()
        .max()
    )

    print()
    print(
        f"[CHECK] max lexical reconstruction error = "
        f"{max_error:.6f}"
    )

    if max_error > 0.05:
        print(
            "[WARNING] Незначительное/существенное расхождение "
            "между реконструированным и production lexical score."
        )
    else:
        print(
            "[OK] Диагностическая декомпозиция lexical score "
            "соответствует production scorer."
        )

    # ------------------------------------------------------------
    # 10. Сводка по top-k
    # ------------------------------------------------------------

    top = df.head(top_k)

    print()
    print("-" * 120)
    print("TOP-K SUMMARY")
    print("-" * 120)

    print(
        f"final score:       "
        f"{top.final_score.min():.3f} .. "
        f"{top.final_score.max():.3f}"
    )

    print(
        f"lexical score:     "
        f"{top.lexical_score.min():.3f} .. "
        f"{top.lexical_score.max():.3f}"
    )

    print(
        f"semantic weighted: "
        f"{top.semantic_weighted.min():.3f} .. "
        f"{top.semantic_weighted.max():.3f}"
    )

    print(
        f"semantic cosine:   "
        f"{top.semantic_cosine.min():.3f} .. "
        f"{top.semantic_cosine.max():.3f}"
    )

    print(
        f"negative penalty:  "
        f"{top.negative_penalty.min():.3f} .. "
        f"{top.negative_penalty.max():.3f}"
    )

    print(
        f"unique documents in top-{top_k}: "
        f"{top.document_id.nunique()}"
    )

    # ------------------------------------------------------------
    # 11. Отдельно показываем top candidates
    # ------------------------------------------------------------

    print()
    print("-" * 120)
    print("TOP CANDIDATES WITH TITLES")
    print("-" * 120)

    for _, row in top.iterrows():
        print(
            f"[{int(row['rank']):3d}] "
            f"final={row['final_score']:.3f} | "
            f"lex={row['lexical_score']:.3f} | "
            f"sem={row['semantic_weighted']:.3f} | "
            f"neg={row['negative_penalty']:.3f} | "
            f"doc={row['document_id']} | "
            f"chunk={row['chunk_id']}"
        )

        print(
            f"      title: {row['title']}"
        )

        print(
            f"      terms: body={int(row['body_term_hits'])}, "
            f"metadata={int(row['metadata_term_hits'])}; "
            f"phrases: body={int(row['phrase_body_hits'])}, "
            f"metadata={int(row['phrase_metadata_hits'])}; "
            f"coverage={row['coverage']:.3f}"
        )

        print(
            f"      evidence={int(row['evidence_hits'])}, "
            f"priority={int(row['source_priority_hits'])}, "
            f"year={row['year_score']:.1f}, "
            f"numeric={row['numeric_score']:.1f}"
        )

        if row["negative_categories"]:
            print(
                f"      negative categories: "
                f"{row['negative_categories']}"
            )

        print(
            f"      preview: {row['text_preview']}"
        )

        print()

    return df

COMMON_PROMPT= """
ОБЩЕЕ
Ты эксперт по ESG-разметке для компаний.

Тебе предоставлены только заранее собранные текстовые фрагменты источников компании.
Запрещено использовать интернет, внешние знания или догадки.
Опирайся СТРОГО на предоставленный текст в блоке "СОБРАННЫЕ ДАННЫЕ С САЙТА".

Нужно вернуть ТОЛЬКО валидный JSON-массив из {n} объектов.
Без markdown, без пояснений, без текста до или после JSON.

Каждый объект соответствует одной Variable:
{var_diff}

Ключи каждого объекта строго:
Company_name, Domains, Variable, Evidence_Type, Evidence_Type_Explanation,
Evidence_Citation, Evidence_Link, Evidence_Date, Freshness, Score, Notes,
LLM, Evaluation_Date, INN.

ПРАВИЛА ЗАПОЛНЕНИЯ БАЗОВЫХ ПОЛЕЙ
Company_name: вставь название компании из вводных данных.
Domains: вставь домен компании из вводных данных.
INN: вставь ИНН компании из вводных данных, если отсутствует — "none".
Variable: один из 16 кодов показателей Stage 1.
Evaluation_Date: дата из вводных данных.
LLM: используемая модель.

ТИПЫ Evidence_Type
Выбери ровно один тип, описывающий главный вид доказательства в ссылке:
none — никакой информации по данной теме не обнаружено.
declaration — общая декларация без проверяемых артефактов.
certificate — внешний сертификат, аттестат, лицензия, декларация соответствия, реестр сертификации или скан сертификата.
KPI — числовые показатели или фактические результаты в таблицах, графиках, тексте или отчётности.
target — формальные цели на будущее с числом и сроком.
policy — утверждённая политика, кодекс, положение, стандарт, регламентирующий документ.
process — описанная процедура, порядок, SLA, регламент работы, механизм обработки, мониторинга или контроля.
role — назначенная роль, должность, орган управления, комитет или ответственное лицо с явным указанием.
report — годовой, нефинансовый, ESG, интегрированный или иной отчёт как документ-агрегатор.
page — тематическая веб-страница, относящаяся к практике, но без формализованного документа, метрик или процедуры.
other — ресс-релиз, новость, блог-пост, презентация или иной материал, если ничего из вышеперечисленного не подходит.

Evidence_Type=other используется только для материала, который описывает конкретное совершённое действие, проект, практику, событие, результат или проверяемый факт по оцениваемой переменной. Если новость или страница содержит только намерение, общую приверженность, лозунг, общую фразу о снижении воздействия или стремление улучшать практики без конкретного действия, результата, процедуры или показателя, используй Evidence_Type=declaration.

Приоритет Evidence_Type:
certificate → KPI → target → policy → process → role → report → page → other → declaration → none.

Годовой, ESG, sustainability, integrated report или databook можно использовать как источник для любой переменной, если цитата из него прямо подтверждает конкретную практику, KPI, политику, процесс, цель или роль по этой переменной. Не требуй отдельной тематической веб-страницы, если отчёт содержит прямое и более сильное evidence. Если в отчёте есть конкретные числовые данные по переменной, выбирай Evidence_Type=KPI, а не report.

Score рассчитывается автоматически:
Evidence_Type=none → Score=0
Evidence_Type=declaration → Score=1
Evidence_Type ∈ {policy, process, certificate, KPI, target, role, report, page, other} и Freshness=0 → Score=2
Evidence_Type ∈ {policy, process, certificate, KPI, target, role, report, page, other} и Freshness=1 → Score=3

Запрещено придумывать специальные шкалы баллов для отдельных показателей. Для всех 30 показателей применяется одна и та же логика Score.

СТРОГОЕ ПРАВИЛО ДОКАЗАТЕЛЬНОСТИ И РЕЛЕВАНТНОСТИ: НЕТ ОТКРЫТОГО И ПРЯМО РЕЛЕВАНТНОГО ИСТОЧНИКА = НЕТ БАЛЛА

Score > 0 возможен только если одновременно выполнены все три условия:
1. Evidence_Citation содержит реальную короткую цитату из предоставленного текста.
2. Evidence_Link взят из URL, который есть рядом с найденным фрагментом.
3. Evidence_Type не равен none.

Если нет точного подтверждения:
Evidence_Type = "none"
Evidence_Type_Explanation = "Информация не найдена в предоставленных источниках"
Evidence_Citation = "none"
Evidence_Link = "none"
Evidence_Date = "none"
Freshness = 0
Score = 0

Notes: 
указывай дополнительные артефакты, которые помогают аудитору понять контекст, но не являются главным доказательством. 
Notes не увеличивают Score. 
Пиши кратко, фактологично и без запятых. 
Если дополнительных замечаний нет, пиши none.

Evidence_Type_Explanation: 
Обязательно указывай полностью, какой именно артефакт был найден. 
Формат: название документа, политики, сертификата, роли, отчёта, страницы или процедуры и, если есть, дата утверждения, номер, год или период. 
Не используй слишком общие формулировки вроде "контакты", "о компании", "раздел сайта".

Evidence_Citation:
это дословная цитата из реально открытой страницы или документа. 
Цитата должна подтверждать главный факт из Evidence_Type_Explanation. 
Цитата должна быть взята только из того первоисточника, который реально открыт и прочитан. 
Запрещено пересказывать текст своими словами вместо цитаты. 
Цитата должна содержать не менее 15 слов. 
Если исходный фрагмент короче 15 слов, выбери более длинный фрагмент вокруг него. 
Если в исходной цитате есть запятые, замени все запятые на точку с запятой ";". 
Если Evidence_Type не равен none, Evidence_Citation обязателен. 
Если релевантная ссылка или документ не открылись или не прочитались из-за ошибки 404, блокировки, пустой заглушки, битого файла, таймаута или иной технической проблемы, в Evidence_Citation нужно написать ровно Failure. 
Если Evidence_Type=none потому что после просмотра реально открытых официальных источников информация по критерию не найдена, в Evidence_Citation пиши none.

Evidence_Link:
Бери только URL из предоставленного контекста.
Не конструируй URL.
Если ссылки нет рядом с фрагментом — "none".

Один и тот же Evidence_Link можно использовать для нескольких переменных только при одновременном выполнении двух условий:
1. для каждой переменной указана отдельная Evidence_Citation;
2. каждая цитата прямо подтверждает именно ту переменную, в строке которой она используется.

Evidence_Date:
Бери дату из DATE, Created, LastModified или явно найденного года в тексте.
Формат:
YYYY-MM-DD
Если только год: YYYY-01-01
Если даты нет: "none".

Freshness:
Freshness=1, если Evidence_Date относится к 2023, 2024, 2025 или более поздним годом.
Freshness=0, если Evidence_Date="none" или дата раньше 2023-01-01.
Freshness может принимать только значения 1 или 0.

Открытый источник НЕ считается прямо релевантным, если он:
- содержит только общие слова об ESG, устойчивом развитии, ответственности, качестве, безопасности, заботе о людях, прозрачности или развитии;
- является общей страницей "О компании", "Устойчивое развитие", "Карьера", "Закупки", "Инвесторам" или "Раскрытие информации" без прямого текста по оцениваемой переменной;
- является новостью о награде, премии, рейтинге, участии в конкурсе, конференции, форуме или выставке без конкретного описания практики по оцениваемой переменной;
- содержит только ссылку, кнопку, пункт меню, footer, quick link или название документа, но сам документ или тематическая страница не открыты и не процитированы;
- подтверждает другую ESG-тему, даже если она находится рядом в той же области ESG;
- используется как proxy evidence, то есть косвенный признак вместо прямого доказательства.

ОБЯЗАТЕЛЬНАЯ ПРОВЕРКА ПЕРЕД ЛЮБЫМ Score > 0 (COMPANY-ACTION TEST)
Перед присвоением Evidence_Type проверь цитату по трём пунктам:
1. Это действие/практика/документ ИМЕННО оцениваемой компании — не закон,
   не регулятор, не муниципалитет, не материнская или сторонняя
   организация, не отрасль в целом.
2. Цитата подтверждает ИМЕННО эту переменную — не соседнюю по смыслу
   переменную того же ESG-блока.
3. Связь с переменной прямая, а не частичная/косвенная/через смежную тему.
Если хотя бы один пункт не пройден — Evidence_Type не выше declaration,
даже если документ формально существует и выглядит содержательным.

Примеры недопустимого proxy evidence:
- новость о включении компании в рейтинг работодателей не доказывает Se2, Se3 или Se5, если в ней нет конкретной информации об условиях труда, льготах, обучении, вовлечённости или внутреннем диалоге;
- общая ESG-страница не доказывает Sc1 или Sc2, если в ней нет прямой информации о системе качества, контроле качества, сертификации качества, безопасности продукции или безопасности услуг для клиентов;
- ссылка на Privacy Policy, Data Protection или Cookies в footer не доказывает Sc3, если сама политика не открыта и не процитирована;
- страница "Контакты" не доказывает Sc4, если в ней нет прямого указания на жалобы, претензии, отзывы, сервисный процесс или качество обслуживания;
- горячая линия по этике не доказывает Se5, если она предназначена для сообщений о коррупции, нарушениях, злоупотреблениях или комплаенс-инцидентах;
- социальные проекты и благотворительность не доказывают So2 или So4, если нет механизма диалога с местными сообществами или управления воздействием деятельности компании на местных жителей.


ФИНАЛЬНОЕ РЕШАЮЩЕЕ ПРАВИЛО:
Score>0 разрешён только тогда, когда в одной и той же строке одновременно есть рабочий Evidence_Link, дословная Evidence_Citation из этого Evidence_Link и прямая содержательная связь этой цитаты с оцениваемой переменной. Во всех остальных случаях Score=0.

ПОКАЗАТЕЛИ STAGE 
{stage_descriptions}

КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ
Если информация спорная, слабая или косвенная — выбирай более низкий Score.
Если нет точного evidence, не завышай оценку.

КОНТРОЛЬ ВЫХОДА
Верни ровно {n} объектов.
Все Variable должны присутствовать один раз.
Не добавляй лишние Variable.
Не добавляй поля сверх заданных ключей.
Все отсутствующие текстовые значения — строка "none".
Freshness и Score — числа."""

STAGE1_SYSTEM_PROMPT = """
ОБЩЕЕ
Ты эксперт по ESG-разметке для компаний.

Тебе предоставлены только заранее собранные текстовые фрагменты источников компании.
Запрещено использовать интернет, внешние знания или догадки.
Опирайся СТРОГО на предоставленный текст в блоке "СОБРАННЫЕ ДАННЫЕ С САЙТА".

Нужно вернуть ТОЛЬКО валидный JSON-массив из 16 объектов.
Без markdown, без пояснений, без текста до или после JSON.

Каждый объект соответствует одной Variable:
E1, E2, E3, E4, E5, E6, E7, Sc1, Sc2, Sc3, Sc4, Se1, Se2, Se3, Se4, Se5.

Ключи каждого объекта строго:
Company_name, Domains, Variable, Evidence_Type, Evidence_Type_Explanation,
Evidence_Citation, Evidence_Link, Evidence_Date, Freshness, Score, Notes,
LLM, Evaluation_Date, INN.

ПРАВИЛА ЗАПОЛНЕНИЯ БАЗОВЫХ ПОЛЕЙ
Company_name: вставь название компании из вводных данных.
Domains: вставь домен компании из вводных данных.
INN: вставь ИНН компании из вводных данных, если отсутствует — "none".
Variable: один из 16 кодов показателей Stage 1.
Evaluation_Date: дата из вводных данных.
LLM: используемая модель.

ТИПЫ Evidence_Type
Выбери ровно один тип, описывающий главный вид доказательства в ссылке:
none — никакой информации по данной теме не обнаружено.
declaration — общая декларация без проверяемых артефактов.
certificate — внешний сертификат, аттестат, лицензия, декларация соответствия, реестр сертификации или скан сертификата.
KPI — числовые показатели или фактические результаты в таблицах, графиках, тексте или отчётности.
target — формальные цели на будущее с числом и сроком.
policy — утверждённая политика, кодекс, положение, стандарт, регламентирующий документ.
process — описанная процедура, порядок, SLA, регламент работы, механизм обработки, мониторинга или контроля.
role — назначенная роль, должность, орган управления, комитет или ответственное лицо с явным указанием.
report — годовой, нефинансовый, ESG, интегрированный или иной отчёт как документ-агрегатор.
page — тематическая веб-страница, относящаяся к практике, но без формализованного документа, метрик или процедуры.
other — ресс-релиз, новость, блог-пост, презентация или иной материал, если ничего из вышеперечисленного не подходит.

Evidence_Type=other используется только для материала, который описывает конкретное совершённое действие, проект, практику, событие, результат или проверяемый факт по оцениваемой переменной. Если новость или страница содержит только намерение, общую приверженность, лозунг, общую фразу о снижении воздействия или стремление улучшать практики без конкретного действия, результата, процедуры или показателя, используй Evidence_Type=declaration.

Приоритет Evidence_Type:
certificate → KPI → target → policy → process → role → report → page → other → declaration → none.

Годовой, ESG, sustainability, integrated report или databook можно использовать как источник для любой переменной, если цитата из него прямо подтверждает конкретную практику, KPI, политику, процесс, цель или роль по этой переменной. Не требуй отдельной тематической веб-страницы, если отчёт содержит прямое и более сильное evidence. Если в отчёте есть конкретные числовые данные по переменной, выбирай Evidence_Type=KPI, а не report.

Score рассчитывается автоматически:
Evidence_Type=none → Score=0
Evidence_Type=declaration → Score=1
Evidence_Type ∈ {policy, process, certificate, KPI, target, role, report, page, other} и Freshness=0 → Score=2
Evidence_Type ∈ {policy, process, certificate, KPI, target, role, report, page, other} и Freshness=1 → Score=3

Запрещено придумывать специальные шкалы баллов для отдельных показателей. Для всех 30 показателей применяется одна и та же логика Score.

СТРОГОЕ ПРАВИЛО ДОКАЗАТЕЛЬНОСТИ И РЕЛЕВАНТНОСТИ: НЕТ ОТКРЫТОГО И ПРЯМО РЕЛЕВАНТНОГО ИСТОЧНИКА = НЕТ БАЛЛА

Score > 0 возможен только если одновременно выполнены все три условия:
1. Evidence_Citation содержит реальную короткую цитату из предоставленного текста.
2. Evidence_Link взят из URL, который есть рядом с найденным фрагментом.
3. Evidence_Type не равен none.

Если нет точного подтверждения:
Evidence_Type = "none"
Evidence_Type_Explanation = "Информация не найдена в предоставленных источниках"
Evidence_Citation = "none"
Evidence_Link = "none"
Evidence_Date = "none"
Freshness = 0
Score = 0

Notes: 
указывай дополнительные артефакты, которые помогают аудитору понять контекст, но не являются главным доказательством. 
Notes не увеличивают Score. 
Пиши кратко, фактологично и без запятых. 
Если дополнительных замечаний нет, пиши none.

Evidence_Type_Explanation: 
Обязательно указывай полностью, какой именно артефакт был найден. 
Формат: название документа, политики, сертификата, роли, отчёта, страницы или процедуры и, если есть, дата утверждения, номер, год или период. 
Не используй слишком общие формулировки вроде "контакты", "о компании", "раздел сайта".

Evidence_Citation:
это дословная цитата из реально открытой страницы или документа. 
Цитата должна подтверждать главный факт из Evidence_Type_Explanation. 
Цитата должна быть взята только из того первоисточника, который реально открыт и прочитан. 
Запрещено пересказывать текст своими словами вместо цитаты. 
Цитата должна содержать не менее 15 слов. 
Если исходный фрагмент короче 15 слов, выбери более длинный фрагмент вокруг него. 
Если в исходной цитате есть запятые, замени все запятые на точку с запятой ";". 
Если Evidence_Type не равен none, Evidence_Citation обязателен. 
Если релевантная ссылка или документ не открылись или не прочитались из-за ошибки 404, блокировки, пустой заглушки, битого файла, таймаута или иной технической проблемы, в Evidence_Citation нужно написать ровно Failure. 
Если Evidence_Type=none потому что после просмотра реально открытых официальных источников информация по критерию не найдена, в Evidence_Citation пиши none.

Evidence_Link:
Бери только URL из предоставленного контекста.
Не конструируй URL.
Если ссылки нет рядом с фрагментом — "none".

Один и тот же Evidence_Link можно использовать для нескольких переменных только при одновременном выполнении двух условий:
1. для каждой переменной указана отдельная Evidence_Citation;
2. каждая цитата прямо подтверждает именно ту переменную, в строке которой она используется.

Evidence_Date:
Бери дату из DATE, Created, LastModified или явно найденного года в тексте.
Формат:
YYYY-MM-DD
Если только год: YYYY-01-01
Если даты нет: "none".

Freshness:
Freshness=1, если Evidence_Date относится к 2023, 2024, 2025 или более поздним годом.
Freshness=0, если Evidence_Date="none" или дата раньше 2023-01-01.
Freshness может принимать только значения 1 или 0.

Открытый источник НЕ считается прямо релевантным, если он:
- содержит только общие слова об ESG, устойчивом развитии, ответственности, качестве, безопасности, заботе о людях, прозрачности или развитии;
- является общей страницей "О компании", "Устойчивое развитие", "Карьера", "Закупки", "Инвесторам" или "Раскрытие информации" без прямого текста по оцениваемой переменной;
- является новостью о награде, премии, рейтинге, участии в конкурсе, конференции, форуме или выставке без конкретного описания практики по оцениваемой переменной;
- содержит только ссылку, кнопку, пункт меню, footer, quick link или название документа, но сам документ или тематическая страница не открыты и не процитированы;
- подтверждает другую ESG-тему, даже если она находится рядом в той же области ESG;
- используется как proxy evidence, то есть косвенный признак вместо прямого доказательства.

ОБЯЗАТЕЛЬНАЯ ПРОВЕРКА ПЕРЕД ЛЮБЫМ Score > 0 (COMPANY-ACTION TEST)
Перед присвоением Evidence_Type проверь цитату по трём пунктам:
1. Это действие/практика/документ ИМЕННО оцениваемой компании — не закон,
   не регулятор, не муниципалитет, не материнская или сторонняя
   организация, не отрасль в целом.
2. Цитата подтверждает ИМЕННО эту переменную — не соседнюю по смыслу
   переменную того же ESG-блока.
3. Связь с переменной прямая, а не частичная/косвенная/через смежную тему.
Если хотя бы один пункт не пройден — Evidence_Type не выше declaration,
даже если документ формально существует и выглядит содержательным.

Примеры недопустимого proxy evidence:
- новость о включении компании в рейтинг работодателей не доказывает Se2, Se3 или Se5, если в ней нет конкретной информации об условиях труда, льготах, обучении, вовлечённости или внутреннем диалоге;
- общая ESG-страница не доказывает Sc1 или Sc2, если в ней нет прямой информации о системе качества, контроле качества, сертификации качества, безопасности продукции или безопасности услуг для клиентов;
- ссылка на Privacy Policy, Data Protection или Cookies в footer не доказывает Sc3, если сама политика не открыта и не процитирована;
- страница "Контакты" не доказывает Sc4, если в ней нет прямого указания на жалобы, претензии, отзывы, сервисный процесс или качество обслуживания;
- горячая линия по этике не доказывает Se5, если она предназначена для сообщений о коррупции, нарушениях, злоупотреблениях или комплаенс-инцидентах;
- социальные проекты и благотворительность не доказывают So2 или So4, если нет механизма диалога с местными сообществами или управления воздействием деятельности компании на местных жителей.


ФИНАЛЬНОЕ РЕШАЮЩЕЕ ПРАВИЛО:
Score>0 разрешён только тогда, когда в одной и той же строке одновременно есть рабочий Evidence_Link, дословная Evidence_Citation из этого Evidence_Link и прямая содержательная связь этой цитаты с оцениваемой переменной. Во всех остальных случаях Score=0.

ПОКАЗАТЕЛИ STAGE 1

E1 - ЭКОЛОГИЧЕСКАЯ ПОЛИТИКА И СИСТЕМА ЭКОЛОГИЧЕСКОГО МЕНЕДЖМЕНТА
Оценивает, есть ли у компании формализованная экологическая политика, система экологического менеджмента, ответственные лица, процедуры управления экологическими рисками или сертификация по экологическому менеджменту.
Ищи: экологическую политику; политику в области охраны окружающей среды; ISO 14001; интегрированную систему менеджмента с экологическим компонентом; разделы отчёта о системе управления экологией; указание экологических подразделений или ответственных лиц.
Выбор Evidence_Type: ISO 14001 или иной внешний сертификат экологического менеджмента → certificate. Формальная экологическая политика или положение → policy. Описанная процедура экологического контроля или управления рисками → process. Назначенный экологический ответственный или подразделение → role. Только общие заявления о заботе об окружающей среде → declaration. Нет информации → none.
Не засчитывай как E1 отдельные данные об отходах, воде, выбросах или энергии, если они не подтверждают именно систему экологического управления.

E2 - КОНТРОЛЬ ЗАГРЯЗНЕНИЙ И ВЫБРОСОВ В ВОЗДУХ, ВОДУ И ПОЧВУ
Оценивает, раскрывает ли компания информацию о выбросах загрязняющих веществ, сбросах, загрязнении почвы, природоохранном контроле, предотвращении аварий и соблюдении природоохранных требований.
Ищи: выбросы загрязняющих веществ в атмосферу; сбросы сточных вод; загрязнение почвы; очистные сооружения; производственный экологический контроль; экологические инциденты; штрафы или соблюдение природоохранных требований; данные о NOx, SOx, пыли, летучих органических соединениях и иных загрязняющих веществах.
Выбор Evidence_Type: количественные показатели выбросов, сбросов или загрязнений → KPI. Цели по снижению загрязняющих выбросов или сбросов с числом и сроком → target. Формальная политика или стандарт по контролю загрязнений → policy. Описанная процедура мониторинга, очистки, контроля или реагирования на инциденты → process. Описание в отчёте без конкретных чисел → report. Тематическая страница без формализованных процедур и метрик → page. Только общие слова "снижаем воздействие" → declaration. Нет информации → none.
Не смешивай E2 с E3. Парниковые газы и климатическая стратегия относятся к E3, если речь идёт о CO2e, Scope 1, Scope 2, Scope 3 или декарбонизации.

E3 - КЛИМАТИЧЕСКАЯ СТРАТЕГИЯ И УПРАВЛЕНИЕ ВЫБРОСАМИ ПАРНИКОВЫХ ГАЗОВ
Оценивает, есть ли у компании раскрытие или управление выбросами парниковых газов, климатическими рисками, углеродным следом, целями декарбонизации или энергоуглеродной стратегией.
Ищи: выбросы CO2e; Scope 1, Scope 2, Scope 3; углеродный след продукции; климатическую стратегию; цели net zero; планы декарбонизации; TCFD; климатические риски; компенсационные проекты; углеродную отчётность.
Выбор Evidence_Type: количественные данные по парниковым газам или углеродному следу → KPI. Цель по снижению GHG или CO2e с числом и сроком → target. Формальная климатическая стратегия или политика → policy. Описанный процесс расчёта, мониторинга или управления GHG → process. Раскрытие в ESG-отчёте или годовом отчёте без конкретных чисел или целей → report. Только общая фраза о борьбе с изменением климата → declaration. Нет информации → none.
Не засчитывай обычное энергосбережение как E3, если нет связи с выбросами парниковых газов, CO2e или климатической повесткой.

E4 - УПРАВЛЕНИЕ ОТХОДАМИ
Оценивает, как компания управляет отходами: образование, сбор, утилизация, переработка, повторное использование, опасные отходы, передача лицензированным операторам, снижение образования отходов.
Ищи: объём отходов; классы опасности отходов; долю переработки; программы раздельного сбора; утилизацию; рециклинг; обращение с опасными отходами; паспорта отходов; договоры с операторами; цели по снижению отходов.
Выбор Evidence_Type: количественные данные по отходам, переработке или утилизации → KPI. Цели по сокращению отходов или увеличению переработки с числом и сроком → target. Политика или стандарт обращения с отходами → policy. Процедура сбора, сортировки, хранения, утилизации или передачи отходов → process. Описание в отчёте без числовых данных → report. Страница о переработке или экологических инициативах без метрик → page. Только общая декларация "сокращаем отходы" → declaration. Нет информации → none.
Не засчитывай как E4 общие слова об экологии, если нет явной связи с отходами.

E5 - ИСПОЛЬЗОВАНИЕ ЭНЕРГИИ И ЭНЕРГОЭФФЕКТИВНОСТЬ
Оценивает, раскрывает ли компания данные о потреблении энергии и мерах по повышению энергоэффективности.
Ищи: потребление электроэнергии; потребление тепловой энергии; потребление топлива; энергоёмкость; энергосбережение; энергоаудит; ISO 50001; модернизацию оборудования; долю возобновляемой энергии; цели по энергоэффективности.
Выбор Evidence_Type: ISO 50001 или внешний сертификат энергоменеджмента → certificate. Количественные показатели потребления энергии или энергоэффективности → KPI. Цели по снижению энергопотребления или энергоёмкости с числом и сроком → target. Политика энергоменеджмента или энергоэффективности → policy. Процедура энергомониторинга, энергоаудита или энергосбережения → process. Описание в отчёте без конкретных чисел → report. Только общие заявления об энергосбережении → declaration. Нет информации → none.
Не подменяй E5 климатическими целями. Если источник говорит только о CO2e без данных об энергии, это скорее E3.

E6 - ИСПОЛЬЗОВАНИЕ ВОДЫ И УПРАВЛЕНИЕ ВОДНЫМИ РЕСУРСАМИ
Оценивает, раскрывает ли компания информацию о водопотреблении, водоотведении, водосбережении, повторном использовании воды и управлении водными рисками.
Ищи: объём водозабора; водопотребление; водоотведение; сточные воды; повторное использование воды; оборотное водоснабжение; очистные сооружения; водные риски; цели по снижению водопотребления.
Выбор Evidence_Type: количественные показатели водопотребления, водоотведения или повторного использования воды → KPI. Цели по снижению водопотребления или увеличению повторного использования воды → target. Политика управления водными ресурсами → policy. Процедуры водного мониторинга, очистки или водосбережения → process. Описание в отчёте без чисел → report. Тематическая страница без метрик и процедур → page. Только общая фраза "бережём воду" → declaration. Нет информации → none.
Не засчитывай в E6 очистку загрязняющих сбросов без связи с водопотреблением или управлением водными ресурсами. Такая информация может относиться к E2.

E7 - БИОРАЗНООБРАЗИЕ И БОЛЕЕ ШИРОКОЕ ВОЗДЕЙСТВИЕ НА ПРИРОДУ
Оценивает, учитывает ли компания влияние на биоразнообразие, экосистемы, леса, почвы, природные территории, животный и растительный мир.
Ищи: программы по сохранению биоразнообразия; рекультивацию земель; восстановление экосистем; лесовосстановление; охраняемые природные территории; оценку воздействия на окружающую среду; компенсационные природоохранные мероприятия; мониторинг флоры и фауны.
Выбор Evidence_Type: количественные показатели по восстановленным территориям, высаженным деревьям, охраняемым видам, площади рекультивации → KPI. Цели по восстановлению природы или биоразнообразию с числом и сроком → target. Политика биоразнообразия или природоохранная политика с явным разделом о природе → policy. Описанный процесс оценки воздействия, рекультивации, мониторинга экосистем → process. Описание в отчёте без чисел → report. Новость или проект по природоохранной инициативе → other. Только общая фраза "заботимся о природе" → declaration. Нет информации → none.
Не засчитывай как E7 обычные данные об отходах, воде или выбросах, если нет связи с экосистемами, биоразнообразием или природными территориями.
Не засчитывай как E7 благополучие сельскохозяйственных животных, ветеринарный контроль, animal welfare на производственных площадках или качество содержания животных, если источник не связывает это прямо с биоразнообразием, дикими видами, экосистемами, природными территориями, воздействием на природу или восстановлением природной среды.

Sc1 - УПРАВЛЕНИЕ КАЧЕСТВОМ ПРОДУКЦИИ ИЛИ УСЛУГ
Оценивает, есть ли у компании формальная система управления качеством продукции или услуг.
Ищи: ISO 9001; отраслевые сертификаты качества; систему менеджмента качества; контроль качества; испытания; лаборатории; стандарты качества; внутренние регламенты; претензионную работу по качеству.
Выбор Evidence_Type: ISO 9001 или другой внешний сертификат качества → certificate. Политика или стандарт качества → policy. Описанный процесс контроля качества, испытаний или аудита качества → process. Количественные показатели качества, брака, рекламаций, удовлетворённости качеством → KPI. Только маркетинговые заявления "высокое качество" → declaration. Нет информации → none.
Не засчитывай как Sc1 обычное описание ассортимента или преимуществ продукта без процедур, сертификатов или проверяемых практик качества.
Сертификация лаборатории, испытательного центра или внутреннего подразделения засчитывается как Sc1 только если источник прямо связывает её с контролем качества продукции или услуг компании для клиентов, потребителей или пользователей. Не засчитывай внутреннюю лабораторную, геологическую, исследовательскую или производственную сертификацию, если в цитате нет связи с качеством продукта или услуги, предоставляемой внешним клиентам.

Sc2 - УПРАВЛЕНИЕ БЕЗОПАСНОСТЬЮ ПРОДУКЦИИ ИЛИ УСЛУГ
Оценивает, как компания обеспечивает безопасность продукции или услуг для клиентов, пользователей и потребителей.
Ищи: сертификаты безопасности; декларации соответствия; технические регламенты; HACCP; ISO 22000; GMP; фармаконадзор; безопасность услуг; инструкции для пользователей; испытания безопасности; отзыв продукции; контроль рисков для потребителей.
Выбор Evidence_Type: внешний сертификат или декларация соответствия безопасности продукции или услуг → certificate. Политика или стандарт безопасности продукции или услуг → policy. Описанный процесс испытаний, контроля безопасности, отзыва продукции или реагирования на инциденты → process. Количественные показатели инцидентов, отзывов, рекламаций по безопасности → KPI. Только общая фраза "безопасная продукция" → declaration. Нет информации → none.
Не смешивай Sc2 с Se1. Безопасность работников - это Se1. Безопасность продукта или услуги для клиента - Sc2.

Sc3 - ДОСТОВЕРНАЯ ИНФОРМАЦИЯ ДЛЯ КЛИЕНТОВ И ЗАЩИТА КЛИЕНТСКИХ ДАННЫХ
Оценивает, раскрывает ли компания правила честного информирования клиентов, маркировки, рекламы, условий использования, а также защиты персональных данных клиентов.
Ищи: политику конфиденциальности; согласие на обработку персональных данных; правила обработки данных; пользовательское соглашение; честную маркировку; раскрытие состава продукции; правила рекламы; информацию о гарантиях; условия оказания услуг; предупреждения о рисках.
Выбор Evidence_Type: формальная политика конфиденциальности, обработки персональных данных, маркировки или раскрытия информации → policy. Описанный процесс обработки данных, запросов субъектов данных, исправления информации или информирования клиентов → process. Назначенный DPO или ответственный за персональные данные → role. Сертификат информационной безопасности, если он относится к защите клиентских данных → certificate. Страница с условиями использования, гарантиями или правилами информирования без формальной политики → page. Только общая фраза "мы уважаем клиентов" → declaration. Нет информации → none.
Обычная страница "Контакты" не считается evidence для Sc3. Горячая линия по этике относится к G3, если её основной фокус - нарушения, а не клиентская информация.

Sc4 - ОБРАТНАЯ СВЯЗЬ, ЖАЛОБЫ КЛИЕНТОВ И СЕРВИСНЫЙ ПРОЦЕСС
Оценивает наличие каналов и процедур для работы с клиентскими обращениями, жалобами, претензиями и удовлетворённостью.
Ищи: формы обратной связи; раздел "жалобы и претензии"; клиентский сервис; контакт-центр; SLA; сроки ответа; порядок рассмотрения обращений; NPS; опросы удовлетворённости; регламент претензионной работы.
Выбор Evidence_Type: описанный процесс обработки жалоб или обращений клиентов → process. Количественные показатели удовлетворённости, NPS, числа обращений, сроков ответа → KPI. Формальная политика клиентского сервиса или претензионной работы → policy. Страница с каналами обратной связи и явным упоминанием отзывов, жалоб или претензий → page. Только общая фраза "мы открыты к обратной связи" → declaration. Нет информации → none.
Не засчитывай обычную страницу "Контакты", если там нет явного указания на жалобы, претензии, отзывы, сервис или качество обслуживания.
Не засчитывай как Sc4 политику конфиденциальности, обработку персональных данных, техническую поддержку сайта, уведомления пользователям или общую форму обратной связи, если в цитате нет прямого указания на жалобы, претензии, обращения клиентов по качеству услуги, отзывы, возвраты, сроки ответа, SLA или порядок рассмотрения клиентских обращений. Защита персональных данных относится к Sc3.
Общее правило для Sc1, Sc2, Sc3 и Sc4: допускается использовать официальные страницы, документы и справочные материалы по конкретным продуктам, приложениям, цифровым сервисам или услугам компании, если они размещены на официальном домене или поддомене компании и прямо подтверждают оцениваемую переменную. Не требуй, чтобы evidence обязательно относилось ко всей компании в целом. Но не засчитывай продуктовую страницу, если она описывает только маркетинговые преимущества продукта без прямого текста о качестве, безопасности, клиентских данных, информировании клиентов, обращениях, жалобах, отзывах, возвратах или сервисном процессе.

Se1 - ОХРАНА ТРУДА И БЕЗОПАСНОСТЬ РАБОТНИКОВ
Оценивает систему охраны труда, промышленной безопасности, предотвращения травматизма и обучения безопасной работе.
Ищи: ISO 45001; OHSAS 18001; политику охраны труда; HSE; промышленную безопасность; обучение безопасности; расследование инцидентов; LTIFR; несчастные случаи; нулевой травматизм.
Выбор Evidence_Type: ISO 45001 или иной внешний сертификат по охране труда → certificate. Количественные показатели травматизма, LTIFR, несчастных случаев → KPI. Цель по снижению травматизма или нулевому травматизму с числом и сроком → target. Политика охраны труда или промышленной безопасности → policy. Описанный процесс инструктажей, расследований, аудитов безопасности → process. Только общие слова "безопасность является приоритетом" → declaration. Нет информации → none.
Не относить сюда безопасность продукции для клиентов. Это Sc2.

Se2 - УСЛОВИЯ ЗАНЯТОСТИ, ОПЛАТА ТРУДА И СОЦИАЛЬНЫЕ ЛЬГОТЫ
Оценивает, раскрывает ли компания информацию об условиях труда, оплате, социальных гарантиях, льготах, ДМС, питании, жилье, транспорте, поддержке семей и иных элементах социального пакета.
Ищи: социальный пакет; ДМС; страхование; компенсации; питание; жильё; транспорт; программы поддержки сотрудников; оплату труда; премии; коллективный договор; условия занятости; гарантии работникам.
Выбор Evidence_Type: формальное положение, коллективный договор или политика оплаты и социальных льгот → policy. Описанный процесс предоставления льгот, компенсаций или социальной поддержки → process. Количественные показатели расходов на персонал, охвата льготами, средней зарплаты → KPI. Страница вакансий или HR-раздел с конкретным перечнем льгот → page. Только общие фразы "достойные условия труда" → declaration. Нет информации → none.
Не засчитывай обучение и развитие сотрудников как Se2, если источник говорит только о курсах или карьерном росте. Это Se3.
Засчитывай как Se2 не только оплату труда и формальный социальный пакет, но и конкретные раскрытые меры социальной поддержки работников и их семей: ДМС, страхование, компенсации, питание, транспорт, жильё, материальную помощь, поддержку родителей, семейные программы, спортивные, оздоровительные и рекреационные мероприятия, если источник содержит конкретное описание, сумму, охват или иной проверяемый факт.

Se3 - ОБУЧЕНИЕ И ПРОФЕССИОНАЛЬНОЕ РАЗВИТИЕ РАБОТНИКОВ
Оценивает, есть ли у компании программы обучения, повышения квалификации, наставничества, кадрового резерва, корпоративного университета и карьерного развития.
Ищи: корпоративный университет; обучение; повышение квалификации; наставничество; кадровый резерв; карьерные треки; индивидуальные планы развития; часы обучения; охват сотрудников обучением; программы для молодых специалистов.
Выбор Evidence_Type: формальная политика или положение об обучении и развитии → policy. Описанные регулярные программы обучения, наставничества, кадрового резерва → process. Количественные показатели часов обучения, числа участников, охвата сотрудников → KPI. Страница HR-раздела с конкретными программами развития → page. Разовая новость о тренинге без признаков системной программы → other. Только общая фраза "развиваем сотрудников" → declaration. Нет информации → none.
Не засчитывай разовую новость о тренинге как системную практику, если нет признаков регулярности.

Se4 - РАЗНООБРАЗИЕ, ИНКЛЮЗИЯ И РАВНЫЕ ВОЗМОЖНОСТИ
Оценивает, раскрывает ли компания политику или практики равных возможностей, недискриминации, гендерного равенства, инклюзии, поддержки людей с инвалидностью или разнообразия персонала.
Ищи: политику недискриминации; равные возможности; diversity and inclusion; гендерное равенство; долю женщин; инклюзивную занятость; адаптацию рабочих мест; поддержку сотрудников с инвалидностью; антидискриминационные нормы в кодексе.
Выбор Evidence_Type: политика равных возможностей, недискриминации или D&I → policy. Описанный процесс предотвращения дискриминации, адаптации рабочих мест или инклюзивного найма → process. Количественные показатели по полу, возрасту, инвалидности, доле женщин в руководстве → KPI. Цели по разнообразию с числом и сроком → target. Норма в кодексе этики о запрете дискриминации → policy. Только общая фраза "уважаем всех сотрудников" → declaration. Нет информации → none.
Не засчитывай общие HR-страницы, если нет явной связи с равными возможностями, недискриминацией или инклюзией.

Se5 - ВОВЛЕЧЁННОСТЬ РАБОТНИКОВ И ВНУТРЕННИЙ ДИАЛОГ
Оценивает, есть ли у компании механизмы внутренней обратной связи, диалога с работниками, опросов вовлечённости, внутренних коммуникаций, участия сотрудников в улучшениях и обсуждении условий труда.
Ищи: опросы вовлечённости; employee engagement; внутренние коммуникации; встречи с руководством; корпоративные порталы; профсоюзы; коллективные переговоры; советы работников; каналы обратной связи; предложения сотрудников; регулярные диалоги с персоналом.
Выбор Evidence_Type: количественные показатели вовлечённости, участия в опросах, числа предложений → KPI. Формальная политика внутреннего диалога, коллективных переговоров или взаимодействия с сотрудниками → policy. Описанный процесс опросов, встреч, сбора предложений или внутренних обращений → process. Страница HR или корпоративной культуры с конкретными каналами диалога → page. Только общие слова "мы ценим мнение сотрудников" → declaration. Нет информации → none.
Не засчитывай горячую линию по нарушениям как Se5, если она предназначена для сообщений о коррупции, этике или злоупотреблениях. Это G3.
Общие опросы заинтересованных сторон, stakeholder surveys, консультации с внешними стейкхолдерами или сбор мнений разных групп не засчитываются как Se5, если в цитате нет отдельного прямого указания на работников, сотрудников, персонал, employee engagement, внутренние коммуникации, внутренний диалог или опрос вовлечённости работников.

КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ
Если информация спорная, слабая или косвенная — выбирай более низкий Score.
Если нет точного evidence, не завышай оценку.

КОНТРОЛЬ ВЫХОДА
Верни ровно 16 объектов.
Все Variable должны присутствовать один раз.
Не добавляй лишние Variable.
Не добавляй поля сверх заданных ключей.
Все отсутствующие текстовые значения — строка "none".
Freshness и Score — числа.
"""


STAGE2_SYSTEM_PROMPT = """
ОБЩЕЕ
Ты эксперт по ESG-разметке для компаний.

Тебе предоставлены только заранее собранные текстовые фрагменты источников компании.
Запрещено использовать интернет, внешние знания или догадки.
Опирайся СТРОГО на предоставленный текст в блоке "СОБРАННЫЕ ДАННЫЕ С САЙТА".

Нужно вернуть ТОЛЬКО валидный JSON-массив из 14 объектов.
Без markdown, без пояснений, без текста до или после JSON.

Каждый объект соответствует одной Variable:
Ss1, Ss2, Ss3, Ss4, So1, So2, So3, So4, G1, G2, G3, G4, G5, G6.

Ключи каждого объекта строго:
Company_name, Domains, Variable, Evidence_Type, Evidence_Type_Explanation,
Evidence_Citation, Evidence_Link, Evidence_Date, Freshness, Score, Notes,
LLM, Evaluation_Date, INN.

ПРАВИЛА ЗАПОЛНЕНИЯ БАЗОВЫХ ПОЛЕЙ
Company_name: вставь название компании из вводных данных.
Domains: вставь домен компании из вводных данных.
INN: вставь ИНН компании из вводных данных, если отсутствует — "none".
Variable: один из 14 кодов показателей Stage 2.
Evaluation_Date: дата из вводных данных.
LLM: используемая модель.

ТИПЫ Evidence_Type
Выбери ровно один тип, описывающий главный вид доказательства в ссылке:
none — никакой информации по данной теме не обнаружено.
declaration — общая декларация без проверяемых артефактов.
certificate — внешний сертификат, аттестат, лицензия, декларация соответствия, реестр сертификации или скан сертификата.
KPI — числовые показатели или фактические результаты в таблицах, графиках, тексте или отчётности.
target — формальные цели на будущее с числом и сроком.
policy — утверждённая политика, кодекс, положение, стандарт, регламентирующий документ.
process — описанная процедура, порядок, SLA, регламент работы, механизм обработки, мониторинга или контроля.
role — назначенная роль, должность, орган управления, комитет или ответственное лицо с явным указанием.
report — годовой, нефинансовый, ESG, интегрированный или иной отчёт как документ-агрегатор.
page — тематическая веб-страница, относящаяся к практике, но без формализованного документа, метрик или процедуры.
other — ресс-релиз, новость, блог-пост, презентация или иной материал, если ничего из вышеперечисленного не подходит.

Evidence_Type=other используется только для материала, который описывает конкретное совершённое действие, проект, практику, событие, результат или проверяемый факт по оцениваемой переменной. Если новость или страница содержит только намерение, общую приверженность, лозунг, общую фразу о снижении воздействия или стремление улучшать практики без конкретного действия, результата, процедуры или показателя, используй Evidence_Type=declaration.

Приоритет Evidence_Type:
certificate → KPI → target → policy → process → role → report → page → other → declaration → none.

Годовой, ESG, sustainability, integrated report или databook можно использовать как источник для любой переменной, если цитата из него прямо подтверждает конкретную практику, KPI, политику, процесс, цель или роль по этой переменной. Не требуй отдельной тематической веб-страницы, если отчёт содержит прямое и более сильное evidence. Если в отчёте есть конкретные числовые данные по переменной, выбирай Evidence_Type=KPI, а не report.

Score рассчитывается автоматически:
Evidence_Type=none → Score=0
Evidence_Type=declaration → Score=1
Evidence_Type ∈ {policy, process, certificate, KPI, target, role, report, page, other} и Freshness=0 → Score=2
Evidence_Type ∈ {policy, process, certificate, KPI, target, role, report, page, other} и Freshness=1 → Score=3

Запрещено придумывать специальные шкалы баллов для отдельных показателей. Для всех 30 показателей применяется одна и та же логика Score.

СТРОГОЕ ПРАВИЛО ДОКАЗАТЕЛЬНОСТИ И РЕЛЕВАНТНОСТИ: НЕТ ОТКРЫТОГО И ПРЯМО РЕЛЕВАНТНОГО ИСТОЧНИКА = НЕТ БАЛЛА

Score > 0 возможен только если одновременно выполнены все три условия:
1. Evidence_Citation содержит реальную короткую цитату из предоставленного текста.
2. Evidence_Link взят из URL, который есть рядом с найденным фрагментом.
3. Evidence_Type не равен none.

Если нет точного подтверждения:
Evidence_Type = "none"
Evidence_Type_Explanation = "Информация не найдена в предоставленных источниках"
Evidence_Citation = "none"
Evidence_Link = "none"
Evidence_Date = "none"
Freshness = 0
Score = 0

Notes: 
указывай дополнительные артефакты, которые помогают аудитору понять контекст, но не являются главным доказательством. 
Notes не увеличивают Score. 
Пиши кратко, фактологично и без запятых. 
Если дополнительных замечаний нет, пиши none.

Evidence_Type_Explanation: 
Обязательно указывай полностью, какой именно артефакт был найден. 
Формат: название документа, политики, сертификата, роли, отчёта, страницы или процедуры и, если есть, дата утверждения, номер, год или период. 
Не используй слишком общие формулировки вроде "контакты", "о компании", "раздел сайта".

Evidence_Citation:
это дословная цитата из реально открытой страницы или документа. 
Цитата должна подтверждать главный факт из Evidence_Type_Explanation. 
Цитата должна быть взята только из того первоисточника, который реально открыт и прочитан. 
Запрещено пересказывать текст своими словами вместо цитаты. 
Цитата должна содержать не менее 15 слов. 
Если исходный фрагмент короче 15 слов, выбери более длинный фрагмент вокруг него. 
Если в исходной цитате есть запятые, замени все запятые на точку с запятой ";". 
Если Evidence_Type не равен none, Evidence_Citation обязателен. 
Если релевантная ссылка или документ не открылись или не прочитались из-за ошибки 404, блокировки, пустой заглушки, битого файла, таймаута или иной технической проблемы, в Evidence_Citation нужно написать ровно Failure. 
Если Evidence_Type=none потому что после просмотра реально открытых официальных источников информация по критерию не найдена, в Evidence_Citation пиши none.

Evidence_Link:
Бери только URL из предоставленного контекста.
Не конструируй URL.
Если ссылки нет рядом с фрагментом — "none".

Один и тот же Evidence_Link можно использовать для нескольких переменных только при одновременном выполнении двух условий:
1. для каждой переменной указана отдельная Evidence_Citation;
2. каждая цитата прямо подтверждает именно ту переменную, в строке которой она используется.

Evidence_Date:
Бери дату из DATE, Created, LastModified или явно найденного года в тексте.
Формат:
YYYY-MM-DD
Если только год: YYYY-01-01
Если даты нет: "none".

Freshness:
Freshness=1, если Evidence_Date относится к 2023, 2024, 2025 или более поздним годом.
Freshness=0, если Evidence_Date="none" или дата раньше 2023-01-01.
Freshness может принимать только значения 1 или 0.

Открытый источник НЕ считается прямо релевантным, если он:
- содержит только общие слова об ESG, устойчивом развитии, ответственности, качестве, безопасности, заботе о людях, прозрачности или развитии;
- является общей страницей "О компании", "Устойчивое развитие", "Карьера", "Закупки", "Инвесторам" или "Раскрытие информации" без прямого текста по оцениваемой переменной;
- является новостью о награде, премии, рейтинге, участии в конкурсе, конференции, форуме или выставке без конкретного описания практики по оцениваемой переменной;
- содержит только ссылку, кнопку, пункт меню, footer, quick link или название документа, но сам документ или тематическая страница не открыты и не процитированы;
- подтверждает другую ESG-тему, даже если она находится рядом в той же области ESG;
- используется как proxy evidence, то есть косвенный признак вместо прямого доказательства.

ОБЯЗАТЕЛЬНАЯ ПРОВЕРКА ПЕРЕД ЛЮБЫМ Score > 0 (COMPANY-ACTION TEST)
Перед присвоением Evidence_Type проверь цитату по трём пунктам:
1. Это действие/практика/документ ИМЕННО оцениваемой компании — не закон,
   не регулятор, не муниципалитет, не материнская или сторонняя
   организация, не отрасль в целом.
2. Цитата подтверждает ИМЕННО эту переменную — не соседнюю по смыслу
   переменную того же ESG-блока.
3. Связь с переменной прямая, а не частичная/косвенная/через смежную тему.
Если хотя бы один пункт не пройден — Evidence_Type не выше declaration,
даже если документ формально существует и выглядит содержательным.

Примеры недопустимого proxy evidence:
- новость о включении компании в рейтинг работодателей не доказывает Se2, Se3 или Se5, если в ней нет конкретной информации об условиях труда, льготах, обучении, вовлечённости или внутреннем диалоге;
- общая ESG-страница не доказывает Sc1 или Sc2, если в ней нет прямой информации о системе качества, контроле качества, сертификации качества, безопасности продукции или безопасности услуг для клиентов;
- ссылка на Privacy Policy, Data Protection или Cookies в footer не доказывает Sc3, если сама политика не открыта и не процитирована;
- страница "Контакты" не доказывает Sc4, если в ней нет прямого указания на жалобы, претензии, отзывы, сервисный процесс или качество обслуживания;
- горячая линия по этике не доказывает Se5, если она предназначена для сообщений о коррупции, нарушениях, злоупотреблениях или комплаенс-инцидентах;
- социальные проекты и благотворительность не доказывают So2 или So4, если нет механизма диалога с местными сообществами или управления воздействием деятельности компании на местных жителей.


ФИНАЛЬНОЕ РЕШАЮЩЕЕ ПРАВИЛО:
Score>0 разрешён только тогда, когда в одной и той же строке одновременно есть рабочий Evidence_Link, дословная Evidence_Citation из этого Evidence_Link и прямая содержательная связь этой цитаты с оцениваемой переменной. Во всех остальных случаях Score=0.

ПОКАЗАТЕЛИ STAGE 2

Ss1 - КОДЕКС ИЛИ ПОЛИТИКА ДЛЯ ПОСТАВЩИКОВ
Оценивает, есть ли у компании формализованные требования к поставщикам, подрядчикам или контрагентам.
Ищи: кодекс поставщика; политику ответственных закупок; стандарт взаимодействия с поставщиками; требования к контрагентам; этические требования в закупках; распространение кодекса этики на поставщиков.
Выбор Evidence_Type: отдельный кодекс поставщика или политика ответственных закупок → policy. Общий кодекс этики, который явно распространяется на поставщиков или контрагентов → policy. Описанная процедура ознакомления поставщиков с требованиями → process. Только общая фраза "работаем с надёжными поставщиками" → declaration. Нет информации → none.
Не засчитывай обычный раздел "Закупки" без требований к поведению, этике, качеству, экологии или социальным аспектам.

Ss2 - ESG-ПРОВЕРКА, ОЦЕНКА И МОНИТОРИНГ ПОСТАВЩИКОВ
Оценивает, проверяет ли компания поставщиков по экологическим, социальным, этическим, антикоррупционным, трудовым или иным ESG-критериям.
Ищи: анкетирование поставщиков; due diligence; проверку контрагентов; ESG-оценку поставщиков; аудиты поставщиков; мониторинг соблюдения кодекса; санкционные проверки; антикоррупционные проверки; экологические и трудовые проверки; риск-ориентированный отбор.
Выбор Evidence_Type: описанный процесс оценки, проверки, аудита или мониторинга поставщиков → process. Формальная политика закупок с ESG-критериями оценки → policy. Количественные показатели числа проверенных поставщиков, аудитов, нарушений → KPI. Анкета или форма самооценки поставщика → process. Только декларация "учитываем ответственность поставщиков" без процедуры → declaration. Нет информации → none.
Не смешивай Ss2 с Ss1. Наличие требований - это Ss1. Проверка их выполнения - Ss2.

Ss3 - СПРАВЕДЛИВЫЕ УСЛОВИЯ И ПРОЗРАЧНОСТЬ ОТНОШЕНИЙ С ПОСТАВЩИКАМИ
Оценивает, раскрывает ли компания принципы прозрачных, конкурентных и справедливых закупок, равного доступа поставщиков, понятных правил тендеров и оплаты.
Ищи: прозрачные закупочные процедуры; электронные торги; равный доступ; недискриминацию поставщиков; сроки оплаты; типовые условия договоров; правила закупок; публикацию тендеров; порядок рассмотрения заявок; жалобы поставщиков.
Выбор Evidence_Type: формальное положение о закупках или правила закупочной деятельности → policy. Описанный процесс тендеров, отбора, рассмотрения заявок, оплаты или жалоб поставщиков → process. Количественные показатели закупок, доли конкурентных процедур, сроков оплаты → KPI. Страница закупок с понятными правилами участия и документами → page. Только общие слова "строим честные отношения" → declaration. Нет информации → none.
Не засчитывай как Ss3 только список текущих тендеров, если нет правил, условий, процедур или принципов прозрачности.

Ss4 - РАЗВИТИЕ ПОСТАВЩИКОВ И ПОДДЕРЖКА ЛОКАЛЬНЫХ ИЛИ МАЛЫХ И СРЕДНИХ ПОСТАВЩИКОВ
Оценивает, поддерживает ли компания развитие поставщиков, локализацию закупок, МСП, местных производителей или повышение устойчивости цепочки поставок.
Ищи: программы развития поставщиков; обучение поставщиков; локальные закупки; поддержку МСП; партнёрские программы; совместные проекты по повышению качества или устойчивости; акселераторы; долгосрочные партнёрства; долю локальных поставщиков.
Выбор Evidence_Type: описанная регулярная программа развития поставщиков или поддержки МСП и локальных поставщиков → process. Политика или стандарт поддержки локальных и МСП-поставщиков → policy. Количественные показатели доли локальных закупок, числа МСП-поставщиков, объёма закупок у локальных поставщиков → KPI. Отдельные новости о поддержке поставщиков без системной программы → other. Только общая фраза "поддерживаем партнёров" → declaration. Нет информации → none.
Не засчитывай обычные закупки у поставщиков как развитие поставщиков, если нет поддержки, обучения, локализации, партнёрской программы или специальных условий.
Доля локальных, региональных, национальных, российских или МСП-поставщиков, а также доля расходов или объёма закупок у таких поставщиков является релевантным KPI для Ss4, если источник прямо раскрывает этот показатель. Не требуй отдельного описания программы развития поставщиков, если есть количественный показатель локализации или поддержки таких поставщиков.
Не засчитывай как Ss4 общие слова о долгосрочных отношениях с поставщиками, партнёрстве, устойчивой цепочке поставок или ответственном подходе к контрагентам, если нет прямого указания на развитие поставщиков, обучение, поддержку МСП, поддержку локальных, региональных или национальных поставщиков, локализацию закупок, специальные условия, партнёрскую программу или количественный показатель закупок у таких поставщиков.

Общее правило для So1, So2, So3 и So4
So1 — социальные проекты, благотворительность, волонтёрство, инвестиции в территории и помощь бенефициарам.
So2 — механизмы диалога с местными жителями и сообществами: слушания, встречи, консультации, приёмные, обращения, жалобы, соглашения с муниципалитетами.
So3 — институциональное взаимодействие с НКО, экспертами, ассоциациями, общественными советами и профессиональными объединениями. Обычная благотворительность через фонд или помощь социальному проекту относится к So1, а не к So3, если нет регулярного партнёрства, экспертного диалога, членства, рабочей группы или общественной инициативы.
So4 — управление негативным воздействием деятельности компании на местные сообщества и их образ жизни: шум, пыль, запахи, транспорт, здоровье жителей, землепользование, переселение, компенсации, жалобы на воздействие, социальные риски. Обычная благотворительность, общественные слушания без описания социального воздействия, ОВОС без социальных мер и обеспечение надёжности клиентских услуг не засчитываются как So4.

So1 - ПОДДЕРЖКА МЕСТНЫХ СООБЩЕСТВ И СОЦИАЛЬНЫЕ ИНВЕСТИЦИИ
Оценивает благотворительность, социальные проекты, инвестиции в территории присутствия, поддержку образования, культуры, спорта, здравоохранения и уязвимых групп.
Ищи: социальные инвестиции; благотворительные программы; развитие территорий; корпоративное волонтёрство; поддержку школ, вузов, больниц, спорта, культуры; фонды; долгосрочные социальные программы; суммы инвестиций и охват.
Выбор Evidence_Type: количественные показатели социальных инвестиций, числа проектов, бенефициаров, волонтёрских часов → KPI. Формальная политика благотворительности или социальных инвестиций → policy. Описанная регулярная программа поддержки местных сообществ → process. Раздел сайта о социальных проектах → page. Описание в ESG-отчёте или годовом отчёте → report. Одна или несколько новостей о социальных проектах → other. Только общая фраза "помогаем обществу" → declaration. Нет информации → none.
Не занижай до declaration, если есть реальные проекты, партнёры, суммы или конкретные территории.
Новость, пресс-релиз или страница о социальных проектах засчитывается как So1, если в ней прямо указаны реальные социальные проекты, территории, партнёры, суммы инвестиций, число бенефициаров, охват, волонтёрские часы или иные проверяемые результаты. Не требуй отдельной политики или отчёта, если новость содержит такие конкретные данные.

So2 - СТРУКТУРИРОВАННЫЙ ДИАЛОГ С МЕСТНЫМИ СООБЩЕСТВАМИ
Оценивает, есть ли у компании регулярные и понятные механизмы диалога с жителями территорий присутствия и местными сообществами.
Ищи: общественные слушания; встречи с жителями; приёмные на территориях; консультации; общественные обсуждения проектов; механизмы жалоб жителей; регулярные диалоговые площадки; соглашения с муниципалитетами; планы взаимодействия с сообществами.
Выбор Evidence_Type: описанный процесс консультаций, слушаний, встреч или рассмотрения обращений местных жителей → process. Политика взаимодействия с местными сообществами или стейкхолдерами → policy. Количественные показатели числа встреч, обращений, консультаций, участников → KPI. Страница о взаимодействии с местными сообществами → page. Описание в отчёте → report. Отдельная новость о встрече или слушаниях → other. Только общая фраза "ведём диалог с обществом" → declaration. Нет информации → none.
Не засчитывай как So2 общую благотворительность без механизма диалога. Это So1.

So3 - ВЗАИМОДЕЙСТВИЕ С НКО, ЭКСПЕРТАМИ, АССОЦИАЦИЯМИ И ОБЩЕСТВЕННЫМИ ИНСТИТУТАМИ
Оценивает участие компании в диалоге с внешними общественными стейкхолдерами за пределами локальных сообществ: НКО, экспертами, профессиональными объединениями, общественными советами, отраслевыми ESG-инициативами.
Ищи: партнёрства с НКО; участие в ассоциациях; общественные и экспертные советы; отраслевые хартии; ESG-коалиции; рабочие группы; взаимодействие с университетами и экспертными организациями; публичные консультации.
Выбор Evidence_Type: формальная политика взаимодействия со стейкхолдерами → policy. Описанный процесс регулярного взаимодействия с НКО, экспертами, ассоциациями → process. Количественные показатели числа партнёров, мероприятий, соглашений, инициатив → KPI. Страница с описанием партнёрств и общественных инициатив → page. Описание в отчёте → report. Отдельная новость о партнёрстве или участии в инициативе → other. Только общая фраза "сотрудничаем с партнёрами" → declaration. Нет информации → none.
Не засчитывай коммерческие выставки, маркетинговые мероприятия или клиентские акции, если нет общественного, экспертного или ESG-содержания.

So4 - УПРАВЛЕНИЕ ВОЗДЕЙСТВИЕМ КОМПАНИИ НА МЕСТНЫЕ СООБЩЕСТВА И ИХ ОБРАЗ ЖИЗНИ
Оценивает, признаёт ли компания своё потенциальное негативное влияние на местные сообщества и управляет ли им: шум, пыль, транспорт, переселение, землепользование, здоровье жителей, традиционный образ жизни, доступ к ресурсам.
Ищи: оценку социального воздействия; управление жалобами жителей; компенсационные меры; снижение шума, пыли, запахов; транспортное воздействие; переселение; воздействие на коренные народы; сохранение традиционного образа жизни; планы управления социальными рисками.
Выбор Evidence_Type: описанный процесс оценки и управления воздействием на местные сообщества → process. Политика по социальному воздействию, правам местных жителей или коренных народов → policy. Количественные показатели жалоб, компенсаций, сниженных воздействий, мониторинга → KPI. Цели по снижению воздействия на местные сообщества с числом и сроком → target. Описание в отчёте → report. Отдельная новость или кейс о снижении воздействия → other. Только общая фраза "учитываем интересы жителей" → declaration. Нет информации → none.
Не засчитывай обычные социальные проекты как So4, если нет связи с управлением воздействием самой деятельности компании на жизнь местных сообществ.

G1 - ИНТЕГРАЦИЯ ESG В МИССИЮ И СТРАТЕГИЮ
Оценивает, встроены ли ESG, устойчивое развитие или корпоративная ответственность в миссию, стратегию, ценности, долгосрочные цели или бизнес-модель компании.
Ищи: ESG-стратегию; стратегию устойчивого развития; миссию с устойчивым развитием; ESG-приоритеты; стратегические цели; интеграцию ESG в бизнес-модель; стратегию до определённого года.
Выбор Evidence_Type: формальный документ стратегии устойчивого развития или ESG-стратегии → policy. Цели ESG или устойчивого развития с числом и сроком → target. Страница о стратегии или ESG-приоритетах → page. Описание в годовом или ESG-отчёте → report. Только одно общее упоминание ESG в миссии или ценностях → declaration. Нет информации → none.
Не засчитывай отдельные экологические или социальные проекты как G1, если они не связаны со стратегией, миссией или долгосрочными приоритетами.

G2 - КОДЕКС ЭТИКИ, АНТИКОРРУПЦИЯ И КОНФЛИКТЫ ИНТЕРЕСОВ
Оценивает наличие формализованных правил деловой этики, антикоррупции, предотвращения конфликта интересов, подарков, взаимодействия с госорганами и контрагентами.
Ищи: кодекс этики; кодекс делового поведения; антикоррупционную политику; конфликт интересов; подарки и представительские расходы; комплаенс; деловую этику; обучение антикоррупции.
Выбор Evidence_Type: кодекс этики, антикоррупционная политика или политика конфликта интересов → policy. Описанный процесс декларирования конфликта интересов, проверки контрагентов, расследований, обучения → process. Количественные показатели обучения, сообщений, расследований, нарушений → KPI. Назначенная комплаенс-функция или ответственный за этику → role. Только общая фраза "мы против коррупции" → declaration. Нет информации → none.
Не засчитывай горячую линию как G2, если источник описывает только канал сообщений без содержания этических или антикоррупционных правил. Это G3.

G3 - КАНАЛЫ СООБЩЕНИЙ О НАРУШЕНИЯХ И ЗАЩИТА ИНФОРМАТОРОВ
Оценивает наличие каналов для сообщений о нарушениях, анонимности, конфиденциальности, защиты от преследования и процедуры рассмотрения сообщений.
Ищи: горячую линию; линию доверия; whistleblowing; сообщения о нарушениях; анонимные обращения; конфиденциальность; защиту заявителей; запрет преследования; порядок рассмотрения обращений.
Выбор Evidence_Type: описанный процесс подачи, рассмотрения и защиты заявителей → process. Политика информирования о нарушениях или защиты заявителей → policy. Страница с телефоном, email или веб-формой для сообщений о нарушениях → page. Количественные показатели обращений, расследований, подтверждённых нарушений → KPI. Назначенный омбудсмен, комплаенс-офицер или ответственный за канал → role. Только общая фраза "можно сообщить о нарушениях" без канала → declaration. Нет информации → none.
Клиентские жалобы на качество или сервис не относятся к G3. Они относятся к Sc4.

G4 - СТРУКТУРА КОРПОРАТИВНОГО УПРАВЛЕНИЯ И РАСПРЕДЕЛЕНИЕ ОТВЕТСТВЕННОСТИ
Оценивает, раскрывает ли компания органы управления, комитеты, распределение полномочий, ответственность за ESG, комплаенс, риски и систему принятия решений.
Ищи: совет директоров; наблюдательный совет; правление; комитеты; ESG-комитет; комитет по аудиту; комитет по устойчивому развитию; распределение ответственности; оргструктуру; положения об органах управления; биографии руководителей.
Выбор Evidence_Type: формальное положение об органах управления, комитетах или корпоративном управлении → policy. Назначенный ESG-комитет, комитет по аудиту, комплаенс-функция или ответственное лицо → role. Описанный процесс корпоративного управления, управления рисками или принятия решений → process. Количественные показатели состава совета, независимости, посещаемости, гендерного состава → KPI. Раздел сайта с органами управления и руководством → page. Описание в годовом отчёте → report. Только общая фраза "эффективное управление" → declaration. Нет информации → none.
Не засчитывай простую страницу "О компании", если там нет органов управления, ролей или распределения ответственности.

G5 - ОТЧЁТНОСТЬ, РАСКРЫТИЕ ИНФОРМАЦИИ И ПРОЗРАЧНОСТЬ
Оценивает, раскрывает ли компания отчётность, ESG-данные, финансовую информацию, структуру собственности, ключевые показатели деятельности и регулярные документы для внешних пользователей.
Ищи: ESG-отчёт; отчёт об устойчивом развитии; годовой отчёт; интегрированный отчёт; финансовую отчётность; раскрытие информации; раздел для инвесторов; структуру собственности; акционеров; бенефициаров; ключевые показатели; архив отчётности.
Выбор Evidence_Type: ESG-отчёт, отчёт об устойчивом развитии, годовой или интегрированный отчёт → report. Таблицы с финансовыми, ESG или операционными KPI → KPI. Формальная политика раскрытия информации → policy. Раздел "Раскрытие информации" или "Инвесторам" с документами и отчётами → page. Описанный процесс раскрытия информации или подготовки отчётности → process. Только общая фраза "мы открыты и прозрачны" → declaration. Нет информации → none.
Если в источнике есть отчёт и числовые показатели, выбирай по главному доказательству. Если ссылка ведёт на сам отчёт как документ-агрегатор, обычно ставь report. Если ссылка ведёт на таблицу KPI, ставь KPI.

G6 - УПРАВЛЕНИЕ РИСКАМИ, ВНУТРЕННИЙ КОНТРОЛЬ И АУДИТ
Оценивает, раскрывает ли компания систему выявления, оценки, мониторинга и снижения рисков, а также связанные с ней механизмы внутреннего контроля, внутреннего аудита, комплаенс-контроля или независимой проверки эффективности контрольных процедур.
Ищи: систему управления рисками; карту рисков; реестр рисков; риск-менеджмент; комитет по рискам; риск-офицера; внутренний контроль; систему внутреннего контроля; внутренний аудит; службу внутреннего аудита; аудиторский комитет; трёхлинейную модель; комплаенс-контроль; процедуры идентификации, оценки, мониторинга, отчётности и снижения рисков; контрольные процедуры; проверки эффективности контроля.
Выбор Evidence_Type: формальная политика, положение или регламент об управлении рисками, внутреннем контроле или внутреннем аудите → policy. Описанный процесс идентификации, оценки, мониторинга, снижения или отчётности по рискам → process. Назначенный комитет по рискам, служба внутреннего аудита, риск-менеджер, риск-офицер, аудиторский комитет или ответственное подразделение → role. Количественные показатели проверок, аудитов, выявленных нарушений, рисковых событий, контрольных мероприятий или устранённых недостатков → KPI. Описание системы управления рисками, внутреннего контроля или внутреннего аудита в годовом, ESG или интегрированном отчёте → report. Страница о риск-менеджменте, внутреннем контроле или внутреннем аудите → page. Только общая фраза "мы управляем рисками" или "контролируем деятельность" → declaration. Нет информации → none.
Не засчитывай как G6 простое перечисление органов управления, совета директоров, правления, руководителей или комитетов без описания системы управления рисками, внутреннего контроля, внутреннего аудита или контрольных процедур. Это относится к G4.
Не засчитывай как G6 обычную финансовую отчётность, раскрытие информации, аудиторское заключение по бухгалтерской отчётности или раздел "Инвесторам", если источник не раскрывает систему управления рисками, внутреннего контроля или внутреннего аудита. Это относится к G5.
Не засчитывай как G6 отдельные экологические, производственные, трудовые, клиентские или поставщицкие риски, если источник не описывает общий корпоративный процесс управления рисками или систему контроля. Такие сведения могут относиться к E, Sc, Se, Ss или So в зависимости от темы.

КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ
Если информация спорная, слабая или косвенная — выбирай более низкий Score.
Если нет точного evidence, не завышай оценку.

КОНТРОЛЬ ВЫХОДА
Верни ровно 14 объектов.
Все Variable должны присутствовать один раз.
Не добавляй лишние Variable.
Не добавляй поля сверх заданных ключей.
Все отсутствующие текстовые значения — строка "none".
Freshness и Score — числа.
"""


VARIABLE_DESCRIPTIONS = {
    "E1": textwrap.dedent("""\
        E1 - ЭКОЛОГИЧЕСКАЯ ПОЛИТИКА И СИСТЕМА ЭКОЛОГИЧЕСКОГО МЕНЕДЖМЕНТА
        Оценивает, есть ли у компании формализованная экологическая политика, система экологического менеджмента, ответственные лица, процедуры управления экологическими рисками или сертификация по экологическому менеджменту.
        Ищи: экологическую политику; политику в области охраны окружающей среды; ISO 14001; интегрированную систему менеджмента с экологическим компонентом; разделы отчёта о системе управления экологией; указание экологических подразделений или ответственных лиц.
        Выбор Evidence_Type: ISO 14001 или иной внешний сертификат экологического менеджмента → certificate. Формальная экологическая политика или положение → policy. Описанная процедура экологического контроля или управления рисками → process. Назначенный экологический ответственный или подразделение → role. Только общие заявления о заботе об окружающей среде → declaration. Нет информации → none.
        Не засчитывай как E1 отдельные данные об отходах, воде, выбросах или энергии, если они не подтверждают именно систему экологического управления.
    """).strip(),

    "E2": textwrap.dedent("""\
        E2 - КОНТРОЛЬ ЗАГРЯЗНЕНИЙ И ВЫБРОСОВ В ВОЗДУХ, ВОДУ И ПОЧВУ
        Оценивает, раскрывает ли компания информацию о выбросах загрязняющих веществ, сбросах, загрязнении почвы, природоохранном контроле, предотвращении аварий и соблюдении природоохранных требований.
        Ищи: выбросы загрязняющих веществ в атмосферу; сбросы сточных вод; загрязнение почвы; очистные сооружения; производственный экологический контроль; экологические инциденты; штрафы или соблюдение природоохранных требований; данные о NOx, SOx, пыли, летучих органических соединениях и иных загрязняющих веществах.
        Выбор Evidence_Type: количественные показатели выбросов, сбросов или загрязнений → KPI. Цели по снижению загрязняющих выбросов или сбросов с числом и сроком → target. Формальная политика или стандарт по контролю загрязнений → policy. Описанная процедура мониторинга, очистки, контроля или реагирования на инциденты → process. Описание в отчёте без конкретных чисел → report. Тематическая страница без формализованных процедур и метрик → page. Только общие слова "снижаем воздействие" → declaration. Нет информации → none.
        Не смешивай E2 с E3. Парниковые газы и климатическая стратегия относятся к E3, если речь идёт о CO2e, Scope 1, Scope 2, Scope 3 или декарбонизации.
    """).strip(),

    "E3": textwrap.dedent("""\
        E3 - КЛИМАТИЧЕСКАЯ СТРАТЕГИЯ И УПРАВЛЕНИЕ ВЫБРОСАМИ ПАРНИКОВЫХ ГАЗОВ
        Оценивает, есть ли у компании раскрытие или управление выбросами парниковых газов, климатическими рисками, углеродным следом, целями декарбонизации или энергоуглеродной стратегией.
        Ищи: выбросы CO2e; Scope 1, Scope 2, Scope 3; углеродный след продукции; климатическую стратегию; цели net zero; планы декарбонизации; TCFD; климатические риски; компенсационные проекты; углеродную отчётность.
        Выбор Evidence_Type: количественные данные по парниковым газам или углеродному следу → KPI. Цель по снижению GHG или CO2e с числом и сроком → target. Формальная климатическая стратегия или политика → policy. Описанный процесс расчёта, мониторинга или управления GHG → process. Раскрытие в ESG-отчёте или годовом отчёте без конкретных чисел или целей → report. Только общая фраза о борьбе с изменением климата → declaration. Нет информации → none.
        Не засчитывай обычное энергосбережение как E3, если нет связи с выбросами парниковых газов, CO2e или климатической повесткой.
    """).strip(),

    "E4": textwrap.dedent("""\
        E4 - УПРАВЛЕНИЕ ОТХОДАМИ
        Оценивает, как компания управляет отходами: образование, сбор, утилизация, переработка, повторное использование, опасные отходы, передача лицензированным операторам, снижение образования отходов.
        Ищи: объём отходов; классы опасности отходов; долю переработки; программы раздельного сбора; утилизацию; рециклинг; обращение с опасными отходами; паспорта отходов; договоры с операторами; цели по снижению отходов.
        Выбор Evidence_Type: количественные данные по отходам, переработке или утилизации → KPI. Цели по сокращению отходов или увеличению переработки с числом и сроком → target. Политика или стандарт обращения с отходами → policy. Процедура сбора, сортировки, хранения, утилизации или передачи отходов → process. Описание в отчёте без числовых данных → report. Страница о переработке или экологических инициативах без метрик → page. Только общая декларация "сокращаем отходы" → declaration. Нет информации → none.
        Не засчитывай как E4 общие слова об экологии, если нет явной связи с отходами.
    """).strip(),

    "E5": textwrap.dedent("""\
        E5 - ИСПОЛЬЗОВАНИЕ ЭНЕРГИИ И ЭНЕРГОЭФФЕКТИВНОСТЬ
        Оценивает, раскрывает ли компания данные о потреблении энергии и мерах по повышению энергоэффективности.
        Ищи: потребление электроэнергии; потребление тепловой энергии; потребление топлива; энергоёмкость; энергосбережение; энергоаудит; ISO 50001; модернизацию оборудования; долю возобновляемой энергии; цели по энергоэффективности.
        Выбор Evidence_Type: ISO 50001 или внешний сертификат энергоменеджмента → certificate. Количественные показатели потребления энергии или энергоэффективности → KPI. Цели по снижению энергопотребления или энергоёмкости с числом и сроком → target. Политика энергоменеджмента или энергоэффективности → policy. Процедура энергомониторинга, энергоаудита или энергосбережения → process. Описание в отчёте без конкретных чисел → report. Только общие заявления об энергосбережении → declaration. Нет информации → none.
        Не подменяй E5 климатическими целями. Если источник говорит только о CO2e без данных об энергии, это скорее E3.
    """).strip(),

    "E6": textwrap.dedent("""\
        E6 - ИСПОЛЬЗОВАНИЕ ВОДЫ И УПРАВЛЕНИЕ ВОДНЫМИ РЕСУРСАМИ
        Оценивает, раскрывает ли компания информацию о водопотреблении, водоотведении, водосбережении, повторном использовании воды и управлении водными рисками.
        Ищи: объём водозабора; водопотребление; водоотведение; сточные воды; повторное использование воды; оборотное водоснабжение; очистные сооружения; водные риски; цели по снижению водопотребления.
        Выбор Evidence_Type: количественные показатели водопотребления, водоотведения или повторного использования воды → KPI. Цели по снижению водопотребления или увеличению повторного использования воды → target. Политика управления водными ресурсами → policy. Процедуры водного мониторинга, очистки или водосбережения → process. Описание в отчёте без чисел → report. Тематическая страница без метрик и процедур → page. Только общая фраза "бережём воду" → declaration. Нет информации → none.
        Не засчитывай в E6 очистку загрязняющих сбросов без связи с водопотреблением или управлением водными ресурсами. Такая информация может относиться к E2.
    """).strip(),

    "E7": textwrap.dedent("""\
        E7 - БИОРАЗНООБРАЗИЕ И БОЛЕЕ ШИРОКОЕ ВОЗДЕЙСТВИЕ НА ПРИРОДУ
        Оценивает, учитывает ли компания влияние на биоразнообразие, экосистемы, леса, почвы, природные территории, животный и растительный мир.
        Ищи: программы по сохранению биоразнообразия; рекультивацию земель; восстановление экосистем; лесовосстановление; охраняемые природные территории; оценку воздействия на окружающую среду; компенсационные природоохранные мероприятия; мониторинг флоры и фауны.
        Выбор Evidence_Type: количественные показатели по восстановленным территориям, высаженным деревьям, охраняемым видам, площади рекультивации → KPI. Цели по восстановлению природы или биоразнообразию с числом и сроком → target. Политика биоразнообразия или природоохранная политика с явным разделом о природе → policy. Описанный процесс оценки воздействия, рекультивации, мониторинга экосистем → process. Описание в отчёте без чисел → report. Новость или проект по природоохранной инициативе → other. Только общая фраза "заботимся о природе" → declaration. Нет информации → none.
        Не засчитывай как E7 обычные данные об отходах, воде или выбросах, если нет связи с экосистемами, биоразнообразием или природными территориями.
        Не засчитывай как E7 благополучие сельскохозяйственных животных, ветеринарный контроль, animal welfare на производственных площадках или качество содержания животных, если источник не связывает это прямо с биоразнообразием, дикими видами, экосистемами, природными территориями, воздействием на природу или восстановлением природной среды.
    """).strip(),

    "Sc1": textwrap.dedent("""\
        Sc1 - УПРАВЛЕНИЕ КАЧЕСТВОМ ПРОДУКЦИИ ИЛИ УСЛУГ
        Оценивает, есть ли у компании формальная система управления качеством продукции или услуг.
        Ищи: ISO 9001; отраслевые сертификаты качества; систему менеджмента качества; контроль качества; испытания; лаборатории; стандарты качества; внутренние регламенты; претензионную работу по качеству.
        Выбор Evidence_Type: ISO 9001 или другой внешний сертификат качества → certificate. Политика или стандарт качества → policy. Описанный процесс контроля качества, испытаний или аудита качества → process. Количественные показатели качества, брака, рекламаций, удовлетворённости качеством → KPI. Только маркетинговые заявления "высокое качество" → declaration. Нет информации → none.
        Не засчитывай как Sc1 обычное описание ассортимента или преимуществ продукта без процедур, сертификатов или проверяемых практик качества.
        Сертификация лаборатории, испытательного центра или внутреннего подразделения засчитывается как Sc1 только если источник прямо связывает её с контролем качества продукции или услуг компании для клиентов, потребителей или пользователей. Не засчитывай внутреннюю лабораторную, геологическую, исследовательскую или производственную сертификацию, если в цитате нет связи с качеством продукта или услуги, предоставляемой внешним клиентам.
    """).strip(),

    "Sc2": textwrap.dedent("""\
        Sc2 - УПРАВЛЕНИЕ БЕЗОПАСНОСТЬЮ ПРОДУКЦИИ ИЛИ УСЛУГ
        Оценивает, как компания обеспечивает безопасность продукции или услуг для клиентов, пользователей и потребителей.
        Ищи: сертификаты безопасности; декларации соответствия; технические регламенты; HACCP; ISO 22000; GMP; фармаконадзор; безопасность услуг; инструкции для пользователей; испытания безопасности; отзыв продукции; контроль рисков для потребителей.
        Выбор Evidence_Type: внешний сертификат или декларация соответствия безопасности продукции или услуг → certificate. Политика или стандарт безопасности продукции или услуг → policy. Описанный процесс испытаний, контроля безопасности, отзыва продукции или реагирования на инциденты → process. Количественные показатели инцидентов, отзывов, рекламаций по безопасности → KPI. Только общая фраза "безопасная продукция" → declaration. Нет информации → none.
        Не смешивай Sc2 с Se1. Безопасность работников - это Se1. Безопасность продукта или услуги для клиента - Sc2.
    """).strip(),

    "Sc3": textwrap.dedent("""\
        Sc3 - ДОСТОВЕРНАЯ ИНФОРМАЦИЯ ДЛЯ КЛИЕНТОВ И ЗАЩИТА КЛИЕНТСКИХ ДАННЫХ
        Оценивает, раскрывает ли компания правила честного информирования клиентов, маркировки, рекламы, условий использования, а также защиты персональных данных клиентов.
        Ищи: политику конфиденциальности; согласие на обработку персональных данных; правила обработки данных; пользовательское соглашение; честную маркировку; раскрытие состава продукции; правила рекламы; информацию о гарантиях; условия оказания услуг; предупреждения о рисках.
        Выбор Evidence_Type: формальная политика конфиденциальности, обработки персональных данных, маркировки или раскрытия информации → policy. Описанный процесс обработки данных, запросов субъектов данных, исправления информации или информирования клиентов → process. Назначенный DPO или ответственный за персональные данные → role. Сертификат информационной безопасности, если он относится к защите клиентских данных → certificate. Страница с условиями использования, гарантиями или правилами информирования без формальной политики → page. Только общая фраза "мы уважаем клиентов" → declaration. Нет информации → none.
        Обычная страница "Контакты" не считается evidence для Sc3. Горячая линия по этике относится к G3, если её основной фокус - нарушения, а не клиентская информация.
        """).strip(),

    "Sc4": textwrap.dedent("""\
        Sc4 - ОБРАТНАЯ СВЯЗЬ, ЖАЛОБЫ КЛИЕНТОВ И СЕРВИСНЫЙ ПРОЦЕСС
        Оценивает наличие каналов и процедур для работы с клиентскими обращениями, жалобами, претензиями и удовлетворённостью.
        Ищи: формы обратной связи; раздел "жалобы и претензии"; клиентский сервис; контакт-центр; SLA; сроки ответа; порядок рассмотрения обращений; NPS; опросы удовлетворённости; регламент претензионной работы.
        Выбор Evidence_Type: описанный процесс обработки жалоб или обращений клиентов → process. Количественные показатели удовлетворённости, NPS, числа обращений, сроков ответа → KPI. Формальная политика клиентского сервиса или претензионной работы → policy. Страница с каналами обратной связи и явным упоминанием отзывов, жалоб или претензий → page. Только общая фраза "мы открыты к обратной связи" → declaration. Нет информации → none.
        Не засчитывай обычную страницу "Контакты", если там нет явного указания на жалобы, претензии, отзывы, сервис или качество обслуживания.
        Не засчитывай как Sc4 политику конфиденциальности, обработку персональных данных, техническую поддержку сайта, уведомления пользователям или общую форму обратной связи, если в цитате нет прямого указания на жалобы, претензии, обращения клиентов по качеству услуги, отзывы, возвраты, сроки ответа, SLA или порядок рассмотрения клиентских обращений. Защита персональных данных относится к Sc3.
        Общее правило для Sc1, Sc2, Sc3 и Sc4: допускается использовать официальные страницы, документы и справочные материалы по конкретным продуктам, приложениям, цифровым сервисам или услугам компании, если они размещены на официальном домене или поддомене компании и прямо подтверждают оцениваемую переменную. Не требуй, чтобы evidence обязательно относилось ко всей компании в целом. Но не засчитывай продуктовую страницу, если она описывает только маркетинговые преимущества продукта без прямого текста о качестве, безопасности, клиентских данных, информировании клиентов, обращениях, жалобах, отзывах, возвратах или сервисном процессе.
    """).strip(),

    "Se1": textwrap.dedent("""\
        Se1 - ОХРАНА ТРУДА И БЕЗОПАСНОСТЬ РАБОТНИКОВ
        Оценивает систему охраны труда, промышленной безопасности, предотвращения травматизма и обучения безопасной работе.
        Ищи: ISO 45001; OHSAS 18001; политику охраны труда; HSE; промышленную безопасность; обучение безопасности; расследование инцидентов; LTIFR; несчастные случаи; нулевой травматизм.
        Выбор Evidence_Type: ISO 45001 или иной внешний сертификат по охране труда → certificate. Количественные показатели травматизма, LTIFR, несчастных случаев → KPI. Цель по снижению травматизма или нулевому травматизму с числом и сроком → target. Политика охраны труда или промышленной безопасности → policy. Описанный процесс инструктажей, расследований, аудитов безопасности → process. Только общие слова "безопасность является приоритетом" → declaration. Нет информации → none.
        Не относить сюда безопасность продукции для клиентов. Это Sc2.
    """).strip(),

    "Se2": textwrap.dedent("""\
        Se2 - УСЛОВИЯ ЗАНЯТОСТИ, ОПЛАТА ТРУДА И СОЦИАЛЬНЫЕ ЛЬГОТЫ
        Оценивает, раскрывает ли компания информацию об условиях труда, оплате, социальных гарантиях, льготах, ДМС, питании, жилье, транспорте, поддержке семей и иных элементах социального пакета.
        Ищи: социальный пакет; ДМС; страхование; компенсации; питание; жильё; транспорт; программы поддержки сотрудников; оплату труда; премии; коллективный договор; условия занятости; гарантии работникам.
        Выбор Evidence_Type: формальное положение, коллективный договор или политика оплаты и социальных льгот → policy. Описанный процесс предоставления льгот, компенсаций или социальной поддержки → process. Количественные показатели расходов на персонал, охвата льготами, средней зарплаты → KPI. Страница вакансий или HR-раздел с конкретным перечнем льгот → page. Только общие фразы "достойные условия труда" → declaration. Нет информации → none.
        Не засчитывай обучение и развитие сотрудников как Se2, если источник говорит только о курсах или карьерном росте. Это Se3.
        Засчитывай как Se2 не только оплату труда и формальный социальный пакет, но и конкретные раскрытые меры социальной поддержки работников и их семей: ДМС, страхование, компенсации, питание, транспорт, жильё, материальную помощь, поддержку родителей, семейные программы, спортивные, оздоровительные и рекреационные мероприятия, если источник содержит конкретное описание, сумму, охват или иной проверяемый факт.
    """).strip(),

    "Se3": textwrap.dedent("""\
        Se3 - ОБУЧЕНИЕ И ПРОФЕССИОНАЛЬНОЕ РАЗВИТИЕ РАБОТНИКОВ
        Оценивает, есть ли у компании программы обучения, повышения квалификации, наставничества, кадрового резерва, корпоративного университета и карьерного развития.
        Ищи: корпоративный университет; обучение; повышение квалификации; наставничество; кадровый резерв; карьерные треки; индивидуальные планы развития; часы обучения; охват сотрудников обучением; программы для молодых специалистов.
        Выбор Evidence_Type: формальная политика или положение об обучении и развитии → policy. Описанные регулярные программы обучения, наставничества, кадрового резерва → process. Количественные показатели часов обучения, числа участников, охвата сотрудников → KPI. Страница HR-раздела с конкретными программами развития → page. Разовая новость о тренинге без признаков системной программы → other. Только общая фраза "развиваем сотрудников" → declaration. Нет информации → none.
        Не засчитывай разовую новость о тренинге как системную практику, если нет признаков регулярности.
    """).strip(),

    "Se4": textwrap.dedent("""\
        Se4 - РАЗНООБРАЗИЕ, ИНКЛЮЗИЯ И РАВНЫЕ ВОЗМОЖНОСТИ
        Оценивает, раскрывает ли компания политику или практики равных возможностей, недискриминации, гендерного равенства, инклюзии, поддержки людей с инвалидностью или разнообразия персонала.
        Ищи: политику недискриминации; равные возможности; diversity and inclusion; гендерное равенство; долю женщин; инклюзивную занятость; адаптацию рабочих мест; поддержку сотрудников с инвалидностью; антидискриминационные нормы в кодексе.
        Выбор Evidence_Type: политика равных возможностей, недискриминации или D&I → policy. Описанный процесс предотвращения дискриминации, адаптации рабочих мест или инклюзивного найма → process. Количественные показатели по полу, возрасту, инвалидности, доле женщин в руководстве → KPI. Цели по разнообразию с числом и сроком → target. Норма в кодексе этики о запрете дискриминации → policy. Только общая фраза "уважаем всех сотрудников" → declaration. Нет информации → none.
        Не засчитывай общие HR-страницы, если нет явной связи с равными возможностями, недискриминацией или инклюзией.
    """).strip(),

    "Se5": textwrap.dedent("""\
        Se5 - ВОВЛЕЧЁННОСТЬ РАБОТНИКОВ И ВНУТРЕННИЙ ДИАЛОГ
        Оценивает, есть ли у компании механизмы внутренней обратной связи, диалога с работниками, опросов вовлечённости, внутренних коммуникаций, участия сотрудников в улучшениях и обсуждении условий труда.
        Ищи: опросы вовлечённости; employee engagement; внутренние коммуникации; встречи с руководством; корпоративные порталы; профсоюзы; коллективные переговоры; советы работников; каналы обратной связи; предложения сотрудников; регулярные диалоги с персоналом.
        Выбор Evidence_Type: количественные показатели вовлечённости, участия в опросах, числа предложений → KPI. Формальная политика внутреннего диалога, коллективных переговоров или взаимодействия с сотрудниками → policy. Описанный процесс опросов, встреч, сбора предложений или внутренних обращений → process. Страница HR или корпоративной культуры с конкретными каналами диалога → page. Только общие слова "мы ценим мнение сотрудников" → declaration. Нет информации → none.
        Не засчитывай горячую линию по нарушениям как Se5, если она предназначена для сообщений о коррупции, этике или злоупотреблениях. Это G3.
        Общие опросы заинтересованных сторон, stakeholder surveys, консультации с внешними стейкхолдерами или сбор мнений разных групп не засчитываются как Se5, если в цитате нет отдельного прямого указания на работников, сотрудников, персонал, employee engagement, внутренние коммуникации, внутренний диалог или опрос вовлечённости работников.
    """).strip(),

    "Ss1": textwrap.dedent("""\                   
        Ss1 - КОДЕКС ИЛИ ПОЛИТИКА ДЛЯ ПОСТАВЩИКОВ
        Оценивает, есть ли у компании формализованные требования к поставщикам, подрядчикам или контрагентам.
        Ищи: кодекс поставщика; политику ответственных закупок; стандарт взаимодействия с поставщиками; требования к контрагентам; этические требования в закупках; распространение кодекса этики на поставщиков.
        Выбор Evidence_Type: отдельный кодекс поставщика или политика ответственных закупок → policy. Общий кодекс этики, который явно распространяется на поставщиков или контрагентов → policy. Описанная процедура ознакомления поставщиков с требованиями → process. Только общая фраза "работаем с надёжными поставщиками" → declaration. Нет информации → none.
        Не засчитывай обычный раздел "Закупки" без требований к поведению, этике, качеству, экологии или социальным аспектам.
    """).strip(),
                           
    "Ss2": textwrap.dedent("""\
        Ss2 - ESG-ПРОВЕРКА, ОЦЕНКА И МОНИТОРИНГ ПОСТАВЩИКОВ
        Оценивает, проверяет ли компания поставщиков по экологическим, социальным, этическим, антикоррупционным, трудовым или иным ESG-критериям.
        Ищи: анкетирование поставщиков; due diligence; проверку контрагентов; ESG-оценку поставщиков; аудиты поставщиков; мониторинг соблюдения кодекса; санкционные проверки; антикоррупционные проверки; экологические и трудовые проверки; риск-ориентированный отбор.
        Выбор Evidence_Type: описанный процесс оценки, проверки, аудита или мониторинга поставщиков → process. Формальная политика закупок с ESG-критериями оценки → policy. Количественные показатели числа проверенных поставщиков, аудитов, нарушений → KPI. Анкета или форма самооценки поставщика → process. Только декларация "учитываем ответственность поставщиков" без процедуры → declaration. Нет информации → none.
        Не смешивай Ss2 с Ss1. Наличие требований - это Ss1. Проверка их выполнения - Ss2.
    """).strip(),

    "Ss3": textwrap.dedent("""\
        Ss3 - СПРАВЕДЛИВЫЕ УСЛОВИЯ И ПРОЗРАЧНОСТЬ ОТНОШЕНИЙ С ПОСТАВЩИКАМИ
        Оценивает, раскрывает ли компания принципы прозрачных, конкурентных и справедливых закупок, равного доступа поставщиков, понятных правил тендеров и оплаты.
        Ищи: прозрачные закупочные процедуры; электронные торги; равный доступ; недискриминацию поставщиков; сроки оплаты; типовые условия договоров; правила закупок; публикацию тендеров; порядок рассмотрения заявок; жалобы поставщиков.
        Выбор Evidence_Type: формальное положение о закупках или правила закупочной деятельности → policy. Описанный процесс тендеров, отбора, рассмотрения заявок, оплаты или жалоб поставщиков → process. Количественные показатели закупок, доли конкурентных процедур, сроков оплаты → KPI. Страница закупок с понятными правилами участия и документами → page. Только общие слова "строим честные отношения" → declaration. Нет информации → none.
        Не засчитывай как Ss3 только список текущих тендеров, если нет правил, условий, процедур или принципов прозрачности.
    """).strip(),

    "Ss4": textwrap.dedent("""\
        Ss4 - РАЗВИТИЕ ПОСТАВЩИКОВ И ПОДДЕРЖКА ЛОКАЛЬНЫХ ИЛИ МАЛЫХ И СРЕДНИХ ПОСТАВЩИКОВ
        Оценивает, поддерживает ли компания развитие поставщиков, локализацию закупок, МСП, местных производителей или повышение устойчивости цепочки поставок.
        Ищи: программы развития поставщиков; обучение поставщиков; локальные закупки; поддержку МСП; партнёрские программы; совместные проекты по повышению качества или устойчивости; акселераторы; долгосрочные партнёрства; долю локальных поставщиков.
        Выбор Evidence_Type: описанная регулярная программа развития поставщиков или поддержки МСП и локальных поставщиков → process. Политика или стандарт поддержки локальных и МСП-поставщиков → policy. Количественные показатели доли локальных закупок, числа МСП-поставщиков, объёма закупок у локальных поставщиков → KPI. Отдельные новости о поддержке поставщиков без системной программы → other. Только общая фраза "поддерживаем партнёров" → declaration. Нет информации → none.
        Не засчитывай обычные закупки у поставщиков как развитие поставщиков, если нет поддержки, обучения, локализации, партнёрской программы или специальных условий.
        Доля локальных, региональных, национальных, российских или МСП-поставщиков, а также доля расходов или объёма закупок у таких поставщиков является релевантным KPI для Ss4, если источник прямо раскрывает этот показатель. Не требуй отдельного описания программы развития поставщиков, если есть количественный показатель локализации или поддержки таких поставщиков.
        Не засчитывай как Ss4 общие слова о долгосрочных отношениях с поставщиками, партнёрстве, устойчивой цепочке поставок или ответственном подходе к контрагентам, если нет прямого указания на развитие поставщиков, обучение, поддержку МСП, поддержку локальных, региональных или национальных поставщиков, локализацию закупок, специальные условия, партнёрскую программу или количественный показатель закупок у таких поставщиков.
    """).strip(),

    "So1": textwrap.dedent("""\
        So1 - ПОДДЕРЖКА МЕСТНЫХ СООБЩЕСТВ И СОЦИАЛЬНЫЕ ИНВЕСТИЦИИ
        Оценивает благотворительность, социальные проекты, инвестиции в территории присутствия, поддержку образования, культуры, спорта, здравоохранения и уязвимых групп.
        Ищи: социальные инвестиции; благотворительные программы; развитие территорий; корпоративное волонтёрство; поддержку школ, вузов, больниц, спорта, культуры; фонды; долгосрочные социальные программы; суммы инвестиций и охват.
        Выбор Evidence_Type: количественные показатели социальных инвестиций, числа проектов, бенефициаров, волонтёрских часов → KPI. Формальная политика благотворительности или социальных инвестиций → policy. Описанная регулярная программа поддержки местных сообществ → process. Раздел сайта о социальных проектах → page. Описание в ESG-отчёте или годовом отчёте → report. Одна или несколько новостей о социальных проектах → other. Только общая фраза "помогаем обществу" → declaration. Нет информации → none.
        Не занижай до declaration, если есть реальные проекты, партнёры, суммы или конкретные территории.
        Новость, пресс-релиз или страница о социальных проектах засчитывается как So1, если в ней прямо указаны реальные социальные проекты, территории, партнёры, суммы инвестиций, число бенефициаров, охват, волонтёрские часы или иные проверяемые результаты. Не требуй отдельной политики или отчёта, если новость содержит такие конкретные данные.
    """).strip(),
                           
    "So2": textwrap.dedent("""\
        So2 - СТРУКТУРИРОВАННЫЙ ДИАЛОГ С МЕСТНЫМИ СООБЩЕСТВАМИ
        Оценивает, есть ли у компании регулярные и понятные механизмы диалога с жителями территорий присутствия и местными сообществами.
        Ищи: общественные слушания; встречи с жителями; приёмные на территориях; консультации; общественные обсуждения проектов; механизмы жалоб жителей; регулярные диалоговые площадки; соглашения с муниципалитетами; планы взаимодействия с сообществами.
        Выбор Evidence_Type: описанный процесс консультаций, слушаний, встреч или рассмотрения обращений местных жителей → process. Политика взаимодействия с местными сообществами или стейкхолдерами → policy. Количественные показатели числа встреч, обращений, консультаций, участников → KPI. Страница о взаимодействии с местными сообществами → page. Описание в отчёте → report. Отдельная новость о встрече или слушаниях → other. Только общая фраза "ведём диалог с обществом" → declaration. Нет информации → none.
        Не засчитывай как So2 общую благотворительность без механизма диалога. Это So1.
    """).strip(),
                           
    "So3": textwrap.dedent("""\
        So3 - ВЗАИМОДЕЙСТВИЕ С НКО, ЭКСПЕРТАМИ, АССОЦИАЦИЯМИ И ОБЩЕСТВЕННЫМИ ИНСТИТУТАМИ
        Оценивает участие компании в диалоге с внешними общественными стейкхолдерами за пределами локальных сообществ: НКО, экспертами, профессиональными объединениями, общественными советами, отраслевыми ESG-инициативами.
        Ищи: партнёрства с НКО; участие в ассоциациях; общественные и экспертные советы; отраслевые хартии; ESG-коалиции; рабочие группы; взаимодействие с университетами и экспертными организациями; публичные консультации.
        Выбор Evidence_Type: формальная политика взаимодействия со стейкхолдерами → policy. Описанный процесс регулярного взаимодействия с НКО, экспертами, ассоциациями → process. Количественные показатели числа партнёров, мероприятий, соглашений, инициатив → KPI. Страница с описанием партнёрств и общественных инициатив → page. Описание в отчёте → report. Отдельная новость о партнёрстве или участии в инициативе → other. Только общая фраза "сотрудничаем с партнёрами" → declaration. Нет информации → none.
        Не засчитывай коммерческие выставки, маркетинговые мероприятия или клиентские акции, если нет общественного, экспертного или ESG-содержания.
    """).strip(),
                           
    "So4": textwrap.dedent("""\
        So4 - УПРАВЛЕНИЕ ВОЗДЕЙСТВИЕМ КОМПАНИИ НА МЕСТНЫЕ СООБЩЕСТВА И ИХ ОБРАЗ ЖИЗНИ
        Оценивает, признаёт ли компания своё потенциальное негативное влияние на местные сообщества и управляет ли им: шум, пыль, транспорт, переселение, землепользование, здоровье жителей, традиционный образ жизни, доступ к ресурсам.
        Ищи: оценку социального воздействия; управление жалобами жителей; компенсационные меры; снижение шума, пыли, запахов; транспортное воздействие; переселение; воздействие на коренные народы; сохранение традиционного образа жизни; планы управления социальными рисками.
        Выбор Evidence_Type: описанный процесс оценки и управления воздействием на местные сообщества → process. Политика по социальному воздействию, правам местных жителей или коренных народов → policy. Количественные показатели жалоб, компенсаций, сниженных воздействий, мониторинга → KPI. Цели по снижению воздействия на местные сообщества с числом и сроком → target. Описание в отчёте → report. Отдельная новость или кейс о снижении воздействия → other. Только общая фраза "учитываем интересы жителей" → declaration. Нет информации → none.
        Не засчитывай обычные социальные проекты как So4, если нет связи с управлением воздействием самой деятельности компании на жизнь местных сообществ.
    """).strip(),
                           
    "G1": textwrap.dedent("""\
        G1 - ИНТЕГРАЦИЯ ESG В МИССИЮ И СТРАТЕГИЮ
        Оценивает, встроены ли ESG, устойчивое развитие или корпоративная ответственность в миссию, стратегию, ценности, долгосрочные цели или бизнес-модель компании.
        Ищи: ESG-стратегию; стратегию устойчивого развития; миссию с устойчивым развитием; ESG-приоритеты; стратегические цели; интеграцию ESG в бизнес-модель; стратегию до определённого года.
        Выбор Evidence_Type: формальный документ стратегии устойчивого развития или ESG-стратегии → policy. Цели ESG или устойчивого развития с числом и сроком → target. Страница о стратегии или ESG-приоритетах → page. Описание в годовом или ESG-отчёте → report. Только одно общее упоминание ESG в миссии или ценностях → declaration. Нет информации → none.
        Не засчитывай отдельные экологические или социальные проекты как G1, если они не связаны со стратегией, миссией или долгосрочными приоритетами.
    """).strip(),
                           
    "G2": textwrap.dedent("""\
        G2 - КОДЕКС ЭТИКИ, АНТИКОРРУПЦИЯ И КОНФЛИКТЫ ИНТЕРЕСОВ
        Оценивает наличие формализованных правил деловой этики, антикоррупции, предотвращения конфликта интересов, подарков, взаимодействия с госорганами и контрагентами.
        Ищи: кодекс этики; кодекс делового поведения; антикоррупционную политику; конфликт интересов; подарки и представительские расходы; комплаенс; деловую этику; обучение антикоррупции.
        Выбор Evidence_Type: кодекс этики, антикоррупционная политика или политика конфликта интересов → policy. Описанный процесс декларирования конфликта интересов, проверки контрагентов, расследований, обучения → process. Количественные показатели обучения, сообщений, расследований, нарушений → KPI. Назначенная комплаенс-функция или ответственный за этику → role. Только общая фраза "мы против коррупции" → declaration. Нет информации → none.
        Не засчитывай горячую линию как G2, если источник описывает только канал сообщений без содержания этических или антикоррупционных правил. Это G3.
    """).strip(),
                           
    "G3": textwrap.dedent("""\
        G3 - КАНАЛЫ СООБЩЕНИЙ О НАРУШЕНИЯХ И ЗАЩИТА ИНФОРМАТОРОВ
        Оценивает наличие каналов для сообщений о нарушениях, анонимности, конфиденциальности, защиты от преследования и процедуры рассмотрения сообщений.
        Ищи: горячую линию; линию доверия; whistleblowing; сообщения о нарушениях; анонимные обращения; конфиденциальность; защиту заявителей; запрет преследования; порядок рассмотрения обращений.
        Выбор Evidence_Type: описанный процесс подачи, рассмотрения и защиты заявителей → process. Политика информирования о нарушениях или защиты заявителей → policy. Страница с телефоном, email или веб-формой для сообщений о нарушениях → page. Количественные показатели обращений, расследований, подтверждённых нарушений → KPI. Назначенный омбудсмен, комплаенс-офицер или ответственный за канал → role. Только общая фраза "можно сообщить о нарушениях" без канала → declaration. Нет информации → none.
        Клиентские жалобы на качество или сервис не относятся к G3. Они относятся к Sc4.
    """).strip(),
                           
    "G4": textwrap.dedent("""\
        G4 - СТРУКТУРА КОРПОРАТИВНОГО УПРАВЛЕНИЯ И РАСПРЕДЕЛЕНИЕ ОТВЕТСТВЕННОСТИ
        Оценивает, раскрывает ли компания органы управления, комитеты, распределение полномочий, ответственность за ESG, комплаенс, риски и систему принятия решений.
        Ищи: совет директоров; наблюдательный совет; правление; комитеты; ESG-комитет; комитет по аудиту; комитет по устойчивому развитию; распределение ответственности; оргструктуру; положения об органах управления; биографии руководителей.
        Выбор Evidence_Type: формальное положение об органах управления, комитетах или корпоративном управлении → policy. Назначенный ESG-комитет, комитет по аудиту, комплаенс-функция или ответственное лицо → role. Описанный процесс корпоративного управления, управления рисками или принятия решений → process. Количественные показатели состава совета, независимости, посещаемости, гендерного состава → KPI. Раздел сайта с органами управления и руководством → page. Описание в годовом отчёте → report. Только общая фраза "эффективное управление" → declaration. Нет информации → none.
        Не засчитывай простую страницу "О компании", если там нет органов управления, ролей или распределения ответственности.
    """).strip(),
                           
    "G5": textwrap.dedent("""\
        G5 - ОТЧЁТНОСТЬ, РАСКРЫТИЕ ИНФОРМАЦИИ И ПРОЗРАЧНОСТЬ
        Оценивает, раскрывает ли компания отчётность, ESG-данные, финансовую информацию, структуру собственности, ключевые показатели деятельности и регулярные документы для внешних пользователей.
        Ищи: ESG-отчёт; отчёт об устойчивом развитии; годовой отчёт; интегрированный отчёт; финансовую отчётность; раскрытие информации; раздел для инвесторов; структуру собственности; акционеров; бенефициаров; ключевые показатели; архив отчётности.
        Выбор Evidence_Type: ESG-отчёт, отчёт об устойчивом развитии, годовой или интегрированный отчёт → report. Таблицы с финансовыми, ESG или операционными KPI → KPI. Формальная политика раскрытия информации → policy. Раздел "Раскрытие информации" или "Инвесторам" с документами и отчётами → page. Описанный процесс раскрытия информации или подготовки отчётности → process. Только общая фраза "мы открыты и прозрачны" → declaration. Нет информации → none.
        Если в источнике есть отчёт и числовые показатели, выбирай по главному доказательству. Если ссылка ведёт на сам отчёт как документ-агрегатор, обычно ставь report. Если ссылка ведёт на таблицу KPI, ставь KPI.
    """).strip(),
                           
    "G6": textwrap.dedent("""\
        G6 - УПРАВЛЕНИЕ РИСКАМИ, ВНУТРЕННИЙ КОНТРОЛЬ И АУДИТ
        Оценивает, раскрывает ли компания систему выявления, оценки, мониторинга и снижения рисков, а также связанные с ней механизмы внутреннего контроля, внутреннего аудита, комплаенс-контроля или независимой проверки эффективности контрольных процедур.
        Ищи: систему управления рисками; карту рисков; реестр рисков; риск-менеджмент; комитет по рискам; риск-офицера; внутренний контроль; систему внутреннего контроля; внутренний аудит; службу внутреннего аудита; аудиторский комитет; трёхлинейную модель; комплаенс-контроль; процедуры идентификации, оценки, мониторинга, отчётности и снижения рисков; контрольные процедуры; проверки эффективности контроля.
        Выбор Evidence_Type: формальная политика, положение или регламент об управлении рисками, внутреннем контроле или внутреннем аудите → policy. Описанный процесс идентификации, оценки, мониторинга, снижения или отчётности по рискам → process. Назначенный комитет по рискам, служба внутреннего аудита, риск-менеджер, риск-офицер, аудиторский комитет или ответственное подразделение → role. Количественные показатели проверок, аудитов, выявленных нарушений, рисковых событий, контрольных мероприятий или устранённых недостатков → KPI. Описание системы управления рисками, внутреннего контроля или внутреннего аудита в годовом, ESG или интегрированном отчёте → report. Страница о риск-менеджменте, внутреннем контроле или внутреннем аудите → page. Только общая фраза "мы управляем рисками" или "контролируем деятельность" → declaration. Нет информации → none.
        Не засчитывай как G6 простое перечисление органов управления, совета директоров, правления, руководителей или комитетов без описания системы управления рисками, внутреннего контроля, внутреннего аудита или контрольных процедур. Это относится к G4.
        Не засчитывай как G6 обычную финансовую отчётность, раскрытие информации, аудиторское заключение по бухгалтерской отчётности или раздел "Инвесторам", если источник не раскрывает систему управления рисками, внутреннего контроля или внутреннего аудита. Это относится к G5.
        Не засчитывай как G6 отдельные экологические, производственные, трудовые, клиентские или поставщицкие риски, если источник не описывает общий корпоративный процесс управления рисками или систему контроля. Такие сведения могут относиться к E, Sc, Se, Ss или So в зависимости от темы.
    """).strip(),
}

def normalize_domain_host(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    if not s or s in ("none", "nan"):
        return ""
    for pref in ("domains:", "domain:"):
        if s.startswith(pref):
            s = s[len(pref):].strip()
            break
    s = s.split("://")[-1]
    s = s.split("/")[0].split(":")[0].strip()
    if s.startswith("www."):
        s = s[4:]
    return s

def _parse_doc_date(date_val):
    s = str(date_val or "").strip()
    if not s or s.lower() == "none":
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        m = re.match(r"(\d{4})", s)
        return date(int(m.group(1)), 1, 1) if m else None

def _dates_conflict(date_a, date_b, file_path_a="", file_path_b="",
                     tolerance_days=RAG_DOC_DEDUPE_DATE_TOLERANCE_DAYS):
    """
    True, если у документов удалось определить разные периоды и они не
    укладываются в допуск.
    Приоритет источника даты: сначала структурная дата (date/created/
    lastmodified из метаданных) — сравнение по дням (tolerance_days).
    Только если структурной даты нет хотя бы у одного документа — fallback
    на год из имени файла (менее точный источник) — сравнение по году.
    Если период не определён ни одним способом хотя бы для одного
    документа — не блокируем (полагаемся только на similarity).
    """
    da, db = _parse_doc_date(date_a), _parse_doc_date(date_b)

    if da is not None and db is not None:
        return abs((da - db).days) > tolerance_days

    ya = da.year if da is not None else _extract_year_from_filename(file_path_a)
    yb = db.year if db is not None else _extract_year_from_filename(file_path_b)

    if ya is not None and yb is not None:
        return ya != yb

    return False

_YEAR_IN_FILENAME_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")

def _extract_year_from_filename(file_path):
    """Fallback: год из имени файла, если структурной даты в метаданных нет.
    Возвращает None, если год в имени не найден или найдено несколько
    разных годов (неоднозначность — лучше не гадать)."""
    fp = str(file_path or "")
    years = set(int(y) for y in _YEAR_IN_FILENAME_RE.findall(fp))
    if len(years) == 1:
        return years.pop()
    return None

def _is_pdf_source(doc: dict) -> bool:
    """PDF-документы исключаются из дедупликации — у них нет языковых
    копий (в отличие от веб-страниц), а структурно похожие PDF (например,
    однотипные лабораторные протоколы) — это разные документы, а не дубли."""
    fp = str(doc.get("file_path", "")).lower()
    return "pdf" in fp

def _is_web_page_source(doc: dict) -> bool:
    """Дедупликация имеет смысл только для веб-страниц (могут существовать
    в нескольких языковых версиях). Любые скачанные документы (PDF, годовые
    отчёты, протоколы, формы — независимо от формата хранения на диске)
    исключаются из сравнения: у них нет языковых копий, а структурное
    сходство (одинаковый шаблон, разные даты/содержание) не означает
    дублирование смысла."""
    fp = str(doc.get("file_path", "")).lower()
    return "_web.txt" in fp or fp.endswith("_web.txt")

def normalize_inn_digits(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan"):
        return ""
    s = s.split(".")[0]
    return re.sub(r"\D", "", s)

def compute_freshness(evidence_date: str) -> int:
    if not evidence_date or str(evidence_date).strip().lower() in ("none", "nan", ""):
        return 0
    try:
        y = int(str(evidence_date)[:4])
        return 1 if y >= 2023 else 0
    except Exception:
        return 0

def compute_score(evidence_type: str, freshness: int) -> int:
    if evidence_type == "none":
        return 0
    if evidence_type == "declaration":
        return 1
    if evidence_type in {"policy", "process", "certificate", "KPI",
                          "target", "role", "report", "page", "other"}:
        return 3 if freshness == 1 else 2
    return 0

def build_failure_rows(company_id, domain, inn, reason: str, model: str = "none") -> pd.DataFrame:
    rows = []
    for v in VARIABLES:
        rows.append({
            "Company_name": str(company_id),
            "Domains": str(domain) if domain else "none",
            "Variable": v,
            "Evidence_Type": "none",
            "Evidence_Type_Explanation": "none",
            "Evidence_Citation": "none",
            "Evidence_Link": "none",
            "Evidence_Date": "none",
            "Freshness": 0,
            "Score": 0,
            "Notes": f"pipeline_status:{reason}",
            "LLM": model,
            "Evaluation_Date": date.today().isoformat(),
            "INN": normalize_inn_digits(inn) or "none",
        })
    return pd.DataFrame(rows, columns=HEADER_COLS)

def extract_inn_from_text(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan"):
        return ""
    m = re.search(r"(?<!\d)(\d{10}|\d{12})(?!\d)", s)
    return m.group(1) if m else ""


def load_sources_index(sources_dir=SOURCES_DIR):
    idx = Path(sources_dir) / "_sources_index.csv"
    if not idx.is_file():
        raise FileNotFoundError(f"Не найден индекс источников: {idx}")

    meta = pd.read_csv(idx, dtype=str).fillna("")
    meta.columns = [str(c).strip() for c in meta.columns]

    if "company_id" not in meta.columns:
        raise ValueError(f"В {idx} нет колонки company_id")
    if "file_path" not in meta.columns:
        raise ValueError(f"В {idx} нет колонки file_path")

    if "domain" not in meta.columns:
        meta["domain"] = ""
    if "inn" not in meta.columns:
        meta["inn"] = ""

    meta["_domain_norm"] = meta["domain"].map(normalize_domain_host)
    meta["_inn_k"] = meta["inn"].map(normalize_inn_digits)

    empty = meta["_inn_k"].eq("")
    meta.loc[empty, "_inn_k"] = meta.loc[empty, "company_id"].map(extract_inn_from_text)

    return meta


def clean_source_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    seen = set()

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            lines.append("")
            continue

        low = raw.lower()
        if len(raw) < 3:
            continue
        if any(x in low for x in NEGATIVE_NAV_MARKERS) and len(raw) < 180:
            continue

        key = re.sub(r"\s+", " ", low)
        if len(key) < 240 and key in seen:
            continue
        if len(key) < 240:
            seen.add(key)

        lines.append(raw)

    return "\n".join(lines).strip()


def read_company_docs(company_sources):
    docs = []

    for m in company_sources:
        fp = str(m.get("file_path", "")).strip()
        if not fp:
            continue

        if not os.path.isabs(fp):
            fp = str(Path(SOURCES_DIR) / fp)

        if not os.path.isfile(fp):
            continue

        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            txt = clean_source_text(f.read())

        if len(txt.strip()) < 150:
            continue

        docs.append({
            **m,
            "text": txt,
            "domain_norm": normalize_domain_host(m.get("domain", "")),
        })

    return docs


def _hard_split_long_text(text, max_chars, overlap):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []

    out = []
    step = max(1, max_chars - overlap)
    for start in range(0, len(text), step):
        part = text[start:start + max_chars].strip()
        if len(part) > 80:
            out.append(part)
        if start + max_chars >= len(text):
            break
    return out


def split_blocks_with_meta(text, max_chars=1600, overlap=220):
    text = clean_source_text(text)

    header_lines = []
    body_lines = []

    for line in text.splitlines():
        low = line.lower().strip()
        if (
            low.startswith("url:")
            or low.startswith("date:")
            or low.startswith("title:")
            or low.startswith("компания:")
            or low.startswith("domain:")
            or low.startswith("inn:")
            or low.startswith("created:")
            or low.startswith("lastmodified:")
        ):
            header_lines.append(line.strip())
        else:
            body_lines.append(line)

    header = "\n".join(header_lines[:14]).strip()
    body = "\n".join(body_lines).strip() if body_lines else text

    raw_paras = [p.strip() for p in re.split(r"\n{2,}", body) if len(p.strip()) > 30]
    paras = []
    for p in raw_paras:
        paras.extend(_hard_split_long_text(p, max_chars=max_chars, overlap=overlap))

    chunks = []
    cur = ""

    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur += ("\n\n" if cur else "") + p
        else:
            if cur:
                chunks.append((header + "\n" + cur).strip() if header else cur.strip())
            cur = (cur[-overlap:] + "\n\n" + p) if overlap and cur else p

    if cur:
        chunks.append((header + "\n" + cur).strip() if header else cur.strip())

    safe = []
    for ch in chunks:
        if len(ch) <= max_chars + 1200:
            safe.append(ch)
        else:
            safe.extend(_hard_split_long_text(ch, max_chars=max_chars, overlap=overlap))

    return safe


def build_all_chunks(company_docs, max_chars_per_chunk=1600):
    chunks = []

    for doc_i, d in enumerate(company_docs):
        doc_text = d["text"]
        url = d.get("url", "") or d.get("source_url", "") or d.get("link", "") or d.get("final_url", "") or ""
        title = d.get("title", "") or d.get("page_title", "") or d.get("page_title_parsed", "") or ""
        date_val = d.get("date", "") or d.get("created", "") or d.get("lastmodified", "") or d.get("last_modified", "") or ""

        for ch_i, ch in enumerate(split_blocks_with_meta(doc_text, max_chars=max_chars_per_chunk)):
            chunk_text = ch
            meta_lines = []
            if url and "url:" not in chunk_text.lower():
                meta_lines.append(f"URL: {url}")
            if date_val and "date:" not in chunk_text.lower():
                meta_lines.append(f"DATE: {date_val}")
            if title and "title:" not in chunk_text.lower():
                meta_lines.append(f"TITLE: {title}")
            if meta_lines:
                chunk_text = "\n".join(meta_lines) + "\n" + chunk_text

            chunks.append({
                "chunk_id": f"{doc_i}_{ch_i}",
                "text": chunk_text,
                "domain": d.get("domain", ""),
                "domain_norm": d.get("domain_norm", ""),
                "url": url,
                "title": title,
                "date": date_val,
                "file_path": d.get("file_path", ""),
                "doc_i": doc_i,
                "chunk_i": ch_i,
            })

    return chunks


def tokenize_query_terms(items):
    joined = " ".join(items).lower().replace("ё", "е")
    return set(re.findall(r"[a-zа-я0-9]{3,}", joined, flags=re.I))


def _norm_text_for_score(text):
    return str(text or "").lower().replace("ё", "е")

def contains_term(text, term):
    """
    Проверка наличия слова/фразы с границами.
    """

    if not term:
        return False

    pattern = (
        r"\b"
        + re.escape(term)
        + r"\b"
    )

    return bool(
        re.search(
            pattern,
            text
        )
    )

# NEGATIVE_QUERIES = build_negative_queries()

def chunk_relevance_score(chunk, var, stage_queries=None):
    text = str(chunk["text"])
    low = _norm_text_for_score(text)
    title_low = _norm_text_for_score(chunk.get("title", ""))
    url_low = _norm_text_for_score(chunk.get("url", ""))
    file_low = _norm_text_for_score(chunk.get("file_path", ""))

    score = 0.0
    queries = stage_queries if stage_queries is not None else ESG_QUERIES[var]
    q_terms = tokenize_query_terms(queries)

    for phrase in queries:
        ph = _norm_text_for_score(phrase)
        if ph and ph in low:
            score += 5.0
        if ph and (ph in title_low or ph in url_low or ph in file_low):
            score += 3.0

    hits = 0
    for t in q_terms:
        if t in low:
            hits += 1
            score += 1.4
        if t in title_low or t in url_low or t in file_low:
            score += 1.1

    if q_terms:
        coverage = hits / max(1, len(q_terms))
        score += 8.0 * coverage

    for _, markers in EVIDENCE_MARKERS.items():
        for marker in markers:
            if _norm_text_for_score(marker) in low:
                score += 0.9

    source_blob = " ".join([low[:4000], title_low, url_low, file_low])
    for marker in SOURCE_PRIORITY_MARKERS:
        if _norm_text_for_score(marker) in source_blob:
            score += 1.0

    if re.search(r"\b20\d{2}\b", low):
        score += 1.5
    if re.search(r"\d+[,.]?\d*\s*(%|тонн|т|м3|м²|квт|квт·ч|руб|млн|млрд)", low):
        score += 1.8

    if "url:" in low or chunk.get("url"):
        score += 1.2
    if "date:" in low or "created:" in low or "lastmodified:" in low or chunk.get("date"):
        score += 0.8
    if "title:" in low or chunk.get("title"):
        score += 0.5

    if len(text) < 220:
        score -= 2.0
    if sum(1 for x in NEGATIVE_NAV_MARKERS if x in low) >= 2 and len(text) < 1500:
        score -= 3.0
    if low.count("http") > 10 and len(text) < 1600:
        score -= 1.5

    return score


def retrieve_and_rerank(chunks, *, var, stage_queries=None, recall_k=36, final_k=16,
                        session=None, scores=None):
    """
    Дедупликация + top-k. Обратно совместим с локальной версией
    (stage_queries передаётся как раньше).

    Быстрый путь — передать session=CompanyRAGSession: score-вектор берётся
    из векторизованного кэша (п.3 + п.4). Если session не передан, индекс
    строится один раз на переданные chunks и считается одна переменная.
    """
    if scores is None:
        if session is not None:
            scores = session.get_scores(var, stage_queries)
        else:
            scores = LexicalScorer(chunks).score(var, stage_queries)

    order = np.argsort(-scores, kind="stable")

    picked = []
    seen = set()
    recall_pool = max(recall_k, final_k)

    for rank, idx in enumerate(order):
        if rank >= recall_pool or len(picked) >= final_k:
            break
        s = float(scores[idx])
        if s <= 0:
            continue
        ch = chunks[int(idx)]
        key_text = re.sub(r"\s+", " ", ch["text"][:700].lower())
        key = hashlib.md5(key_text.encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        picked.append({**ch, "score": s})

    return picked


def _format_chunk_for_context(var, ch, i):
    meta = []
    if ch.get("url"):
        meta.append(f"URL: {ch['url']}")
    if ch.get("date"):
        meta.append(f"DATE: {ch['date']}")
    if ch.get("title"):
        meta.append(f"TITLE: {ch['title']}")
    if ch.get("domain"):
        meta.append(f"DOMAIN: {ch['domain']}")

    meta_txt = "\n".join(meta)
    if meta_txt:
        meta_txt += "\n"

    return (
        f"[{var} chunk {i} | relevance={ch.get('score', 0):.2f}]\n"
        + meta_txt
        + str(ch.get("text", "")).strip()
    ).strip()


def _trim_section_to_budget(section, budget):
    section = str(section or "")
    if len(section) <= budget:
        return section
    return section[:budget] + "\n...[section truncated]"


def build_variable_budget_context(chunks_by_var, variables=None, *, full_context_limit, min_chunks_per_var=1):
    variables = list(variables) if variables is not None else list(VARIABLES)
    usable_budget = int(full_context_limit * 0.92)
    per_var_budget = max(1800, usable_budget // max(1, len(variables)))

    sections = []
    global_seen = set()

    for var in variables:
        selected = chunks_by_var.get(var, []) or []

        if not selected:
            sections.append(f"### {var}\nnone")
            continue

        var_parts = []
        var_len = 0

        for j, ch in enumerate(selected, 1):
            key_text = re.sub(r"\s+", " ", ch["text"][:700].lower())
            key = hashlib.md5(key_text.encode("utf-8", errors="ignore")).hexdigest()
            formatted = _format_chunk_for_context(var, ch, j)

            must_take = len(var_parts) < min_chunks_per_var
            if key in global_seen and not must_take:
                continue

            if not must_take and var_len + len(formatted) > per_var_budget:
                continue

            var_parts.append(formatted)
            var_len += len(formatted)
            global_seen.add(key)

            if var_len >= per_var_budget:
                break

        if not var_parts:
            var_parts.append(_format_chunk_for_context(var, selected[0], 1))

        section = f"### {var}\n" + "\n\n---\n\n".join(var_parts)
        section = _trim_section_to_budget(section, per_var_budget + 800)
        sections.append(section)

    context = "\n\n".join(sections)

    if len(context) > full_context_limit:
        tighter_budget = max(1200, int(full_context_limit * 0.88) // max(1, len(VARIABLES)))
        tight_sections = [_trim_section_to_budget(s, tighter_budget) for s in sections]
        context = "\n\n".join(tight_sections)

    if len(context) > full_context_limit:
        context = context[:full_context_limit] + "\n\n...[context truncated by full_context_limit]"

    return context


def build_context_for_company(
    company_docs,
    *,
    full_context_limit=80000,
    recall_k=36,
    final_k=16,
    max_chars_per_chunk=2000,
    session=None,
):
    total_chars = sum(len(d["text"]) for d in company_docs)

    if total_chars <= full_context_limit:
        context = "\n\n".join(d["text"] for d in company_docs)
        if len(context) > full_context_limit:
            context = context[:full_context_limit] + "\n\n...[context truncated by full_context_limit]"
        return context

    if session is None:
        # ad-hoc вызов без внешнего кэша: сессия всё равно создаётся,
        # но живёт только внутри этого вызова.
        session = CompanyRAGSession("<adhoc>", max_chars_per_chunk=max_chars_per_chunk)
        session._docs = company_docs

    chunks = session.get_chunks()

    chunks_by_var = {}
    for var in VARIABLES:
        chunks_by_var[var] = retrieve_and_rerank(
            chunks,
            var=var,
            recall_k=recall_k,
            final_k=final_k,
            session=session,
        )

    return build_variable_budget_context(
        chunks_by_var,
        full_context_limit=full_context_limit,
        min_chunks_per_var=1,
    )


def build_context_for_stage(
    company_docs,
    stage_variables,
    stage_queries,
    *,
    full_context_limit=80000,
    recall_k=36,
    final_k=16,
    max_chars_per_chunk=2000,
    precomputed_chunks=None,
    session=None,
):
    """
    Подготовка контекста для одного этапа.
    Основа подготовки данных и retrieval — из rag_by_variable.py.
    Для каждого Variable используются его lexical + semantic scores,
    если передана CompanyRAGSession.
    """
    total_chars = sum(len(d["text"]) for d in company_docs)

    if total_chars <= full_context_limit:
        print(f"    [context] {total_chars:,} <= full_context_limit={full_context_limit:,} "              
              f"-> retrieval пропущен, весь текст передаётся целиком (без lexical/semantic)")
        context = "\n\n".join(d["text"] for d in company_docs)
        if len(context) > full_context_limit:
            context = context[:full_context_limit] + "\n\n...[context truncated by full_context_limit]"
        return context

    if precomputed_chunks is not None:
        chunks = precomputed_chunks
    elif session is not None:
        chunks = session.get_chunks(company_docs)
    else:
        chunks = build_all_chunks(company_docs, max_chars_per_chunk=max_chars_per_chunk)

    semantic_on = bool(session is not None and getattr(session, "enable_semantic", False))    
    print(f"    [context] {total_chars:,} > full_context_limit={full_context_limit:,} "          f"-> retrieval: chunks={len(chunks)}, vars={len(stage_variables)}, "          f"lexical=on, semantic={'on' if semantic_on else 'off'}")

    chunks_by_var = {}
    for var in stage_variables:
        queries = stage_queries.get(var, ESG_QUERIES.get(var, []))
        chunks_by_var[var] = retrieve_and_rerank(
            chunks,
            var=var,
            stage_queries=queries,
            recall_k=recall_k,
            final_k=final_k,
            session=session,
        )

    return build_variable_budget_context(
        chunks_by_var,
        stage_variables,
        full_context_limit=full_context_limit,
        min_chunks_per_var=1,
    )


# def build_stage_prompt(company_id, domain, inn, context, stage_num, stage_variables):
#     """Строит промпт для конкретного этапа (Stage 1 или Stage 2)."""
#     match stage_num:
#         case "1":
#             var_diff=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "Sc1", "Sc2", "Sc3", "Sc4", "Se1", "Se2", "Se3", "Se4", "Se"]
#         case "2":
#             var_diff=["Ss1", "Ss2", "Ss3", "Ss4", "So1", "So2", "So3", "So4", "G1", "G2", "G3", "G4", "G5", "G6"]

#     system_prompt = COMMON_PROMPT.format(
#             n=len(var_diff),
#             var_diff=", ".join(var_diff),
#             stage_descriptions=_build_stage_descriptions(", ".join(var_diff)),
#         )
#     return f"""{system_prompt}
# ВВОДНЫЕ:
# Company_name: {company_id}
# Domains: {domain}
# INN: {inn}
# Evaluation_Date: {date.today().isoformat()}

# СПИСОК VARIABLE ДЛЯ ОЦЕНКИ (Stage {stage_num}):
# {", ".join(stage_variables)}

# СОБРАННЫЕ ДАННЫЕ С САЙТА:
# {context}

# ВЕРНИ ТОЛЬКО JSON-МАССИВ ИЗ {len(stage_variables)} ОБЪЕКТОВ.
# """

def build_stage_prompt(company_id, domain, inn, context, stage_num, stage_variables, system_prompt):
    """Строит промпт для конкретного этапа (Stage 1 или Stage 2)."""
    return f"""{system_prompt}

ВВОДНЫЕ:
Company_name: {company_id}
Domains: {domain}
INN: {inn}
Evaluation_Date: {date.today().isoformat()}

СПИСОК VARIABLE ДЛЯ ОЦЕНКИ (Stage {stage_num}):
{", ".join(stage_variables)}

СОБРАННЫЕ ДАННЫЕ С САЙТА:
{context}

ВЕРНИ ТОЛЬКО JSON-МАССИВ ИЗ {len(stage_variables)} ОБЪЕКТОВ.
"""

def _post_with_retry(payload, *, max_retries=3, backoff=5.0):
    """
    POST в gateway с повторами при сетевых сбоях/таймаутах/5xx.
    4xx (неверный ключ, битый запрос) — не повторяем, падаем сразу.
    """
    headers = {"Content-Type": "application/json"}
    if CLIENT_API_KEY:
        headers["X-API-Key"] = CLIENT_API_KEY

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{GATEWAY_URL}/v1/generate",
                headers=headers,
                json=payload,
                timeout=GATEWAY_TIMEOUT,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt == max_retries:
                raise
            sleep_s = backoff * attempt  # 5s, 10s, 15s...
            print(f"    [gateway retry] сетевая ошибка ({type(e).__name__}), "
                  f"попытка {attempt}/{max_retries}, жду {sleep_s:.0f}s")
            time.sleep(sleep_s)
            continue

        if response.status_code >= 500:
            last_exc = requests.exceptions.HTTPError(
                f"Gateway HTTP {response.status_code}: {response.text[:300]}"
            )
            if attempt == max_retries:
                response.raise_for_status()
            sleep_s = backoff * attempt
            print(f"    [gateway retry] HTTP {response.status_code}, "
                  f"попытка {attempt}/{max_retries}, жду {sleep_s:.0f}s")
            time.sleep(sleep_s)
            continue

        # 200 или 4xx — не повторяем
        response.raise_for_status()
        return response

    raise last_exc


def call_qwen(prompt, max_output_tokens=16000, temperature=0.0, role=GATEWAY_ROLE):
    payload = {
        "role": role,
        "prompt": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    response = _post_with_retry(payload)
    data = response.json()

    text = data.get("text", "")
    if not text:
        raise ValueError("Gateway вернул пустой text")

    model_name = data.get("model", "unknown")
    return text, model_name


def extract_json_array(text):
    text = str(text).strip()

    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        raise ValueError("В ответе нет JSON-массива")

    return json.loads(m.group(0))


def normalize_output_row(obj, *, company_id, domain, inn, model):
    row = {}
    for k in HEADER_COLS:
        v = obj.get(k, "none") if isinstance(obj, dict) else "none"
        if k in ("Freshness", "Score"):
            try:
                v = int(float(str(v).replace(",", ".")))
            except Exception:
                v = 0
        else:
            if v is None or str(v).strip() == "":
                v = "none"
            else:
                v = str(v).replace("\n", " ").replace(";", ",").strip()
        row[k] = v
    row["Company_name"] = row["Company_name"] if row["Company_name"] != "none" else str(company_id)
    row["Domains"] = row["Domains"] if row["Domains"] != "none" else str(domain)
    row["INN"] = normalize_inn_digits(row["INN"]) or normalize_inn_digits(inn) or "none"
    row["LLM"] = model
    row["Evaluation_Date"] = date.today().isoformat()
    if row["Variable"] not in VARIABLES:
        row["Variable"] = "none"

    model_score = row["Score"]
    model_freshness = row["Freshness"]

    row["Freshness"] = compute_freshness(row["Evidence_Date"])
    row["Score"] = compute_score(row["Evidence_Type"], row["Freshness"])

    if model_score != row["Score"] or model_freshness != row["Freshness"]:
        correction_note = (
            f"score_corrected: model_score={model_score}->{row['Score']}, "
            f"model_freshness={model_freshness}->{row['Freshness']}"
        )
        row["Notes"] = (
            correction_note if row["Notes"] in ("none", "", "nan")
            else f"{row['Notes']} | {correction_note}"
        )
        print(correction_note)
    return row


def parse_json_rating(raw_text, *, company_id, domain, inn, stage_variables, model):
    data = extract_json_array(raw_text)
    if not isinstance(data, list):
        raise ValueError("Ответ должен быть JSON-массивом")
    rows = [
        normalize_output_row(obj, company_id=company_id, domain=domain, inn=inn, model=model)
        for obj in data
        if isinstance(obj, dict)
    ]
    df = pd.DataFrame(rows, columns=HEADER_COLS)
    df = df[df["Variable"].isin(stage_variables)].drop_duplicates("Variable", keep="first")
    existing = set(df["Variable"])
    missing = [v for v in stage_variables if v not in existing]
    filler = []
    for v in missing:
        filler.append({
            "Company_name": str(company_id),
            "Domains": str(domain),
            "Variable": v,
            "Evidence_Type": "none",
            "Evidence_Type_Explanation": "none",
            "Evidence_Citation": "none",
            "Evidence_Link": "none",
            "Evidence_Date": "none",
            "Freshness": 0,
            "Score": 0,
            "Notes": "auto-filled: missing variable in model response",
            "LLM": model,
            "Evaluation_Date": date.today().isoformat(),
            "INN": normalize_inn_digits(inn) or "none",
        })
    if filler:
        df = pd.concat([df, pd.DataFrame(filler)], ignore_index=True)
    order = {v: i for i, v in enumerate(stage_variables)}
    df["_ord"] = df["Variable"].map(order)
    df = df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    return df[HEADER_COLS]


def validate_evidence_links(df, context):
    out = df.copy()
    ctx_low = context.lower()

    notes = []
    for _, r in out.iterrows():
        note = str(r.get("Notes", "none"))
        link = str(r.get("Evidence_Link", "none")).strip().lower()

        if link and link not in ("none", "nan"):
            if link not in ctx_low:
                suffix = "warning: Evidence_Link not found in context"
                note = suffix if note in ("none", "", "nan") else note + " | " + suffix

        notes.append(note if note else "none")

    out["Notes"] = notes
    return out

def flag_reused_evidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mask = (
        df["Evidence_Link"].ne("none")
        & df["Evidence_Citation"].ne("none")
        & df["Evidence_Citation"].ne("Failure")
    )

    dupe_mask = df[mask].duplicated(subset=["Evidence_Link", "Evidence_Citation"], keep=False)
    dupe_idx = df[mask].index[dupe_mask]

    for idx in dupe_idx:
        note = str(df.at[idx, "Notes"])
        warning = "warning: identical Evidence_Link+Citation reused across variables"
        df.at[idx, "Notes"] = warning if note in ("none", "", "nan") else f"{note} | {warning}"

    return df

def get_company_ids(meta_df):
    company_ids = sorted(meta_df["company_id"].dropna().astype(str).unique())

    if RAG_COMPANY_FILTER:
        wanted = {x.strip() for x in RAG_COMPANY_FILTER.split(",") if x.strip()}
        company_ids = [x for x in company_ids if x in wanted]

    if RAG_MAX_COMPANIES:
        try:
            company_ids = company_ids[:int(RAG_MAX_COMPANIES)]
        except Exception:
            pass

    return company_ids


def _build_stage_descriptions(var_diff: str) -> str:
    """
    Собрать блок описаний переменных для промпта на основе строки
    var_diff вида "G5,G6,G7" — только те переменные, что реально
    есть в текущем расхождении оценок.
    """
    codes = [v.strip() for v in var_diff.split(", ") if v.strip()]
    blocks = []
    missing = []
    for code in codes:
        desc = VARIABLE_DESCRIPTIONS.get(code)
        if desc:
            blocks.append(desc)
        else:
            missing.append(code)

    if missing:
        print(f"[WARN] Нет описания для переменных: {', '.join(missing)}")

    return "\n\n".join(blocks)


VALIDATOR_PROMPT_PARTIAL_TEMPLATE = """
ОБЩЕЕ
Ты оцениваешь показатели ESG компании на основе предоставленных тебе заранее спарсенных текстовых блоков (веб-страниц и PDF-документов). Для каждого из показателей ESG нужно заполнить одну строку в выходном файле CSV по правилам, перечисленным ниже.
Тебе запрещено выдумывать информацию, искать ее в интернете или использовать свои внутренние знания о компании. Опирайся СТРОГО на предоставленный текст в блоках "СОБРАННЫЕ ДАННЫЕ С САЙТА".

Также тебе переданы два независимых JSON-массива с оценкой одной и той же компании, но только для тех переменных, по которым было различие в оценке.

ОЦЕНКА 1 (scorer):
{raw_1}

ОЦЕНКА 2 (auditor):
{raw_2}

СОБРАННЫЕ ДАННЫЕ С САЙТА:
{context}

Нужно вернуть ТОЛЬКО валидный JSON-массив из {n} объектов.
Без markdown, без пояснений, без текста до или после JSON.

Каждый объект соответствует одной Variable:
{var_diff}

Ключи каждого объекта строго:
Company_name, Domains, Variable, Evidence_Type, Evidence_Type_Explanation,
Evidence_Citation, Evidence_Link, Evidence_Date, Freshness, Score, Notes,
LLM, Evaluation_Date, INN.

ПРАВИЛА ЗАПОЛНЕНИЯ БАЗОВЫХ ПОЛЕЙ
Company_name: вставь название компании из вводных данных.
Domains: вставь домен компании из вводных данных.
INN: вставь ИНН компании из вводных данных, если отсутствует — "none".
Variable: один из 16 кодов показателей Stage 1.
Evaluation_Date: дата из вводных данных.
LLM: используемая модель.

ТИПЫ Evidence_Type
Выбери ровно один тип, описывающий главный вид доказательства в ссылке:
Если даты нет: "none".

Freshness:
Freshness=1, если Evidence_Date относится к 2023, 2024, 2025 или более поздним годом.
Freshness=0, если Evidence_Date="none" или дата раньше 2023-01-01.
Freshness может принимать только значения 1 или 0.

Открытый источник НЕ считается прямо релевантным, если он:
- содержит только общие слова об ESG, устойчивом развитии, ответственности, качестве, безопасности, заботе о людях, прозрачности или развитии;
- является общей страницей "О компании", "Устойчивое развитие", "Карьера", "Закупки", "Инвесторам" или "Раскрытие информации" без прямого текста по оцениваемой переменной;
- является новостью о награде, премии, рейтинге, участии в конкурсе, конференции, форуме или выставке без конкретного описания практики по оцениваемой переменной;
- содержит только ссылку, кнопку, пункт меню, footer, quick link или название документа, но сам документ или тематическая страница не открыты и не процитированы;
- подтверждает другую ESG-тему, даже если она находится рядом в той же области ESG;
- используется как proxy evidence, то есть косвенный признак вместо прямого доказательства.

Примеры недопустимого proxy evidence:
- новость о включении компании в рейтинг работодателей не доказывает Se2, Se3 или Se5, если в ней нет конкретной информации об условиях труда, льготах, обучении, вовлечённости или внутреннем диалоге;
- общая ESG-страница не доказывает Sc1 или Sc2, если в ней нет прямой информации о системе качества, контроле качества, сертификации качества, безопасности продукции или безопасности услуг для клиентов;
- ссылка на Privacy Policy, Data Protection или Cookies в footer не доказывает Sc3, если сама политика не открыта и не процитирована;
- страница "Контакты" не доказывает Sc4, если в ней нет прямого указания на жалобы, претензии, отзывы, сервисный процесс или качество обслуживания;
- горячая линия по этике не доказывает Se5, если она предназначена для сообщений о коррупции, нарушениях, злоупотреблениях или комплаенс-инцидентах;
- социальные проекты и благотворительность не доказывают So2 или So4, если нет механизма диалога с местными сообществами или управления воздействием деятельности компании на местных жителей.

ОБЯЗАТЕЛЬНАЯ ПРОВЕРКА ПЕРЕД ЛЮБЫМ Score > 0 (COMPANY-ACTION TEST)
Перед присвоением Evidence_Type проверь цитату по трём пунктам:
1. Это действие/практика/документ ИМЕННО оцениваемой компании — не закон,
   не регулятор, не муниципалитет, не материнская или сторонняя
   организация, не отрасль в целом.
2. Цитата подтверждает ИМЕННО эту переменную — не соседнюю по смыслу
   переменную того же ESG-блока.
3. Связь с переменной прямая, а не частичная/косвенная/через смежную тему.
Если хотя бы один пункт не пройден — Evidence_Type не выше declaration,
даже если документ формально существует и выглядит содержательным.

ФИНАЛЬНОЕ РЕШАЮЩЕЕ ПРАВИЛО:
Score>0 разрешён только тогда, когда в одной и той же строке одновременно есть рабочий Evidence_Link, дословная Evidence_Citation из этого Evidence_Link и прямая содержательная связь этой цитаты с оцениваемой переменной. Во всех остальных случаях Score=0.

ПОКАЗАТЕЛИ STAGE 
{stage_descriptions}

КРИТИЧЕСКОЕ ОГРАНИЧЕНИЕ
Если информация спорная, слабая или косвенная — выбирай более низкий Score.
Если нет точного evidence, не завышай оценку.

КОНТРОЛЬ ВЫХОДА
Верни ровно {n} объектов.
Все Variable должны присутствовать один раз.
Не добавляй лишние поля сверх: Company_name, Domains, Variable, Evidence_Type, Evidence_Type_Explanation, Evidence_Citation, Evidence_Link, Evidence_Date, Freshness, Score, Notes, LLM, Evaluation_Date, INN.
Все отсутствующие текстовые значения — строка "none".
Freshness и Score — числа.
"""


def limit_context_chars(context, limit):
    context = str(context or "")
    if len(context) > limit:
        return context[:limit] + "\n\n...[context truncated by full_context_limit]"
    return context


def call_validator_partial(raw_1: str, raw_2: str, var_diff: list, context: str) -> tuple:
    """Точечный validator только по переменным с различающимся Score."""
    prompt = VALIDATOR_PROMPT_PARTIAL_TEMPLATE.format(
        raw_1=raw_1,
        raw_2=raw_2,
        context=context,
        n=len(var_diff),
        var_diff=", ".join(var_diff),
        stage_descriptions=_build_stage_descriptions(", ".join(var_diff)),
    )

    payload = {
        "role": "validator",
        "prompt": prompt,
        "temperature": 0.0,
        "max_output_tokens": max(4000, 800 * len(var_diff)),
    }

    response = _post_with_retry(payload)
    data = response.json()

    text = data.get("text", "")
    if not text:
        raise ValueError("Gateway вернул пустой text")

    return text, data.get("model", "unknown")


def _diff_variables_by_score(df_1: pd.DataFrame, df_2: pd.DataFrame, variables=VARIABLES) -> list:
    """Список переменных, где Score в df_1 и df_2 расходится."""
    s1 = df_1.set_index("Variable")["Score"]
    s2 = df_2.set_index("Variable")["Score"]
    s1 = pd.to_numeric(s1, errors="coerce").fillna(0).astype(int)
    s2 = pd.to_numeric(s2, errors="coerce").fillna(0).astype(int)
    return [v for v in variables if v in s1.index and v in s2.index and int(s1[v]) != int(s2[v])]


def _partial_validate_variables(company_id, domain, inn, df_1, df_2, var_diff, context, safe_company=None):
    """
    Вызывает точечный validator по var_diff, передавая реальный текст
    источников (context). При ошибке падает обратно на результат scorer
    по этим же переменным.
    """
    df_1_idx = df_1.set_index("Variable", drop=False)
    df_2_idx = df_2.set_index("Variable", drop=False)

    records_1 = df_1_idx.loc[var_diff, HEADER_COLS].to_dict("records")
    records_2 = df_2_idx.loc[var_diff, HEADER_COLS].to_dict("records")
    raw_1_diff = json.dumps(records_1, ensure_ascii=False, indent=2)
    raw_2_diff = json.dumps(records_2, ensure_ascii=False, indent=2)

    try:
        raw_final, model_val = call_validator_partial(raw_1_diff, raw_2_diff, var_diff, context)
        print(f"  Validator модель: {model_val}")
        if safe_company:
            (Path(INTERMEDIATE_DIR) / f"{safe_company}_validator_partial_raw.txt").write_text(
                raw_final, encoding="utf-8"
            )
        return parse_json_rating(
            raw_final, company_id=company_id, domain=domain, inn=inn, stage_variables=var_diff, model=model_val
        )
    except Exception as e:
        print(f"  Validator ERROR: {type(e).__name__}: {str(e)[:400]}")
        print("  Fallback по расходящимся переменным: берём результат scorer")
        return df_1_idx.loc[var_diff, HEADER_COLS].reset_index(drop=True)


def resolve_company_scores(
    company_id, domain, inn, df_1: pd.DataFrame, df_2: pd.DataFrame, docs,
    *,
    precomputed_chunks=None,
    full_context_limit=80000,
    recall_k=48,
    final_k=20,
    max_chars_per_chunk=1600,
    safe_company=None,
) -> pd.DataFrame:
    """
    Свести scorer (df_1) и auditor (df_2) в один итоговый DataFrame.
    Расхождения по Score решает точечный validator — с реальным текстом
    источников (только по расходящимся переменным). Совпадающие строки
    берутся из scorer как есть. Если расхождений нет вообще — итог = df_1.
    """
    var_diff = _diff_variables_by_score(df_1, df_2)
    print(f"  Validator: расхождений по Score = {len(var_diff)} из {len(VARIABLES)}")

    if not var_diff:
        print("  Расхождений нет — итог = результат scorer (согласовано с auditor).")
        return df_1.copy().reset_index(drop=True)

    context = build_context_for_stage(
        docs, var_diff, ESG_QUERIES,
        full_context_limit=full_context_limit,
        recall_k=recall_k, final_k=final_k,
        max_chars_per_chunk=max_chars_per_chunk,
        precomputed_chunks=precomputed_chunks,
    )
    context = limit_context_chars(context, full_context_limit)

    df_resolved = _partial_validate_variables(
        company_id, domain, inn, df_1, df_2, var_diff, context, safe_company=safe_company,
    )

    df_1_idx = df_1.set_index("Variable", drop=False)
    unchanged_vars = [v for v in VARIABLES if v not in var_diff and v in df_1_idx.index]
    df_unchanged = (
        df_1_idx.loc[unchanged_vars, HEADER_COLS].reset_index(drop=True)
        if unchanged_vars else pd.DataFrame(columns=HEADER_COLS)
    )

    df_company = pd.concat([df_unchanged, df_resolved], ignore_index=True)
    order = {v: i for i, v in enumerate(VARIABLES)}
    df_company["_ord"] = df_company["Variable"].map(order)
    df_company = df_company.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    return df_company[HEADER_COLS]


def run_scorer_full(
    company_id,
    domain,
    inn,
    docs,
    role_name,
    safe_company,
    full_context_limit=80000,
    recall_k=48,
    final_k=20,
    max_chars_per_chunk=1600,
    precomputed_chunks=None,
    session=None,
):
    """
    Один scorer/auditor проходит ОБА этапа:
    Stage 1 (16) -> Stage 2 (14) -> объединение в 30 строк.
    Возвращает DataFrame из 30 строк и raw JSON для validator.
    """
    print(f"    {role_name}: запуск двухэтапной оценки...")

    context1 = build_context_for_stage(
        docs, STAGE1_VARIABLES, STAGE1_QUERIES,
        full_context_limit=full_context_limit,
        recall_k=recall_k, final_k=final_k,
        max_chars_per_chunk=max_chars_per_chunk,
        precomputed_chunks=precomputed_chunks,
        session=session,
    )
    context1 = limit_context_chars(context1, full_context_limit)
    # исходный вариант
    prompt1 = build_stage_prompt(
        company_id, domain, inn, context1, 1,
        STAGE1_VARIABLES, STAGE1_SYSTEM_PROMPT
    )
    # обощенный вариант
    # prompt1 = build_stage_prompt(
    #         company_id=company_id, domain=domain, inn=inn, context=context1, 
    #         stage_num=1, stage_variables=STAGE1_VARIABLES
    #     )

    (Path(INTERMEDIATE_DIR) / f"{safe_company}_{role_name}_stage1_prompt.txt").write_text(
        prompt1, encoding="utf-8"
    )

    print(f"    {role_name} Stage 1: prompt={len(prompt1):,} chars")
    raw_stage1, model_stage1 = call_qwen(prompt1, 16000, 0.0, role_name)
    (Path(INTERMEDIATE_DIR) / f"{safe_company}_{role_name}_stage1_raw.txt").write_text(
        raw_stage1, encoding="utf-8"
    )
    print(f"    {role_name} Stage 1: model={model_stage1}")

    df_stage1 = parse_json_rating(
        raw_stage1,
        company_id=company_id,
        domain=domain,
        inn=inn,
        stage_variables=STAGE1_VARIABLES,
        model=model_stage1,
    )
    df_stage1 = validate_evidence_links(df_stage1, context1)
    df_stage1.to_csv(
        Path(INTERMEDIATE_DIR) / f"{safe_company}_{role_name}_stage1.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )

    context2 = build_context_for_stage(
        docs, STAGE2_VARIABLES, STAGE2_QUERIES,
        full_context_limit=full_context_limit,
        recall_k=recall_k, final_k=final_k,
        max_chars_per_chunk=max_chars_per_chunk,
        precomputed_chunks=precomputed_chunks,
        session=session,
    )
    context2 = limit_context_chars(context2, full_context_limit)
    prompt2 = build_stage_prompt(
        company_id, domain, inn, context2, 2,
        STAGE2_VARIABLES, STAGE2_SYSTEM_PROMPT
    )

    print(f"    {role_name} Stage 2: prompt={len(prompt2):,} chars")
    raw_stage2, model_stage2 = call_qwen(prompt2, 16000, 0.0, role_name)
    (Path(INTERMEDIATE_DIR) / f"{safe_company}_{role_name}_stage2_raw.txt").write_text(
        raw_stage2, encoding="utf-8"
    )
    print(f"    {role_name} Stage 2: model={model_stage2}")

    df_stage2 = parse_json_rating(
        raw_stage2,
        company_id=company_id,
        domain=domain,
        inn=inn,
        stage_variables=STAGE2_VARIABLES,
        model=model_stage2,
    )
    df_stage2 = validate_evidence_links(df_stage2, context2)
    df_stage2.to_csv(
        Path(INTERMEDIATE_DIR) / f"{safe_company}_{role_name}_stage2.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )

    df_full = pd.concat([df_stage1, df_stage2], ignore_index=True)
    order = {v: i for i, v in enumerate(VARIABLES)}
    df_full["_ord"] = df_full["Variable"].map(order)
    df_full = df_full.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    df_full.to_csv(
        Path(INTERMEDIATE_DIR) / f"{safe_company}_{role_name}_full30.csv",
        sep=";", index=False, encoding="utf-8-sig",
    )

    raw_json = json.dumps(df_full.to_dict("records"), ensure_ascii=False, indent=2)

    print(
        f"    {role_name}: итого {len(df_full)} строк "
        f"(Stage1={len(df_stage1)}, Stage2={len(df_stage2)})"
    )
    return df_full, raw_json


def run_rag_by_variable(
    meta_df=None,
    output_csv=OUTPUT_CSV,
    *,
    full_context_limit=80000,
    recall_k=48,
    final_k=20,
    max_chars_per_chunk=1600,
    sleep_s=0.0,
):
    if meta_df is None:
        meta_df = load_sources_index(SOURCES_DIR)

    company_ids = get_company_ids(meta_df)

    print("=" * 80)
    print("Variable-budget Improved RAG — Two-Stage Evaluation")
    print("=" * 80)
    print("Компаний к прогону:", len(company_ids))
    print("output:", output_csv)
    print(
        f"params: full_context_limit={full_context_limit}, recall_k={recall_k}, "
        f"final_k={final_k}, max_chars_per_chunk={max_chars_per_chunk}"
    )
    print(f"Stage 1: {len(STAGE1_VARIABLES)} переменных (E, Sc, Se)")
    print(f"Stage 2: {len(STAGE2_VARIABLES)} переменных (Ss, So, G)")
    print("\nАрхитектура:")
    print("  Scorer  -> Stage1 + Stage2 -> 30 строк")
    print("  Auditor -> Stage1 + Stage2 -> 30 строк")
    print("  Validator -> только расхождения Score -> финальные 30 строк")
    print("=" * 80)

    all_results = []

    for i, company_id in enumerate(company_ids, 1):
        company_sources = meta_df[
            meta_df["company_id"].astype(str) == company_id
        ].to_dict("records")

        if not company_sources:
            continue

        domain = company_sources[0].get("domain", "none") or "none"
        inn = (
            normalize_inn_digits(company_sources[0].get("inn", ""))
            or extract_inn_from_text(company_id)
            or "none"
        )

        print(f"\n[{i}/{len(company_ids)}]  {company_id} | domain={domain} | inn={inn}")

        session = CompanyRAGSession(
            company_id,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        docs = session.get_docs(company_sources)

        if not docs:
            print("  пропуск: нет пригодных текстов -> записываем failure-строки")
            all_results.append(
                build_failure_rows(company_id, domain, inn, "no_valid_docs", model=GATEWAY_ROLE)
            )
            continue

        total_chars = sum(len(d["text"]) for d in docs)
        if total_chars <= full_context_limit:
            precomputed_chunks = None
        else:
            precomputed_chunks = session.get_chunks(company_sources)
            print(f"    Чанков построено: {len(precomputed_chunks)}")

        safe_company = re.sub(
            r"[^a-zA-Zа-яА-Я0-9_.-]+", "_", company_id
        )[:80]

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_1 = executor.submit(
                    run_scorer_full,
                    company_id, domain, inn, docs, "scorer", safe_company,
                    full_context_limit, recall_k, final_k, max_chars_per_chunk,
                    precomputed_chunks, session,
                )
                future_2 = executor.submit(
                    run_scorer_full,
                    company_id, domain, inn, docs, "auditor", safe_company,
                    full_context_limit, recall_k, final_k, max_chars_per_chunk,
                    precomputed_chunks, session,
                )

                df_1, raw_1 = future_1.result()
                df_2, raw_2 = future_2.result()

        except Exception as e:
            print(
                f"  ERROR при выполнении scorer'ов: "
                f"{type(e).__name__}: {str(e)[:400]}"
            )
            continue

        try:
            df_company = resolve_company_scores(
                company_id, domain, inn, df_1, df_2, docs,
                precomputed_chunks=precomputed_chunks,
                full_context_limit=full_context_limit,
                recall_k=recall_k,
                final_k=final_k,
                max_chars_per_chunk=max_chars_per_chunk,
                safe_company=safe_company,
            )
            df_company = validate_evidence_links(df_company, raw_1 + raw_2)
            df_company = flag_reused_evidence(df_company)   
        except Exception as e:
            print(f"  Validator ERROR: {type(e).__name__}: {str(e)[:400]}")
            print("  Fallback: используем результат scorer")
            df_company = df_1.copy()

        df_company.to_csv(
            Path(INTERMEDIATE_DIR) / f"{safe_company}_final.csv",
            sep=";", index=False, encoding="utf-8-sig",
        )

        all_results.append(df_company)

        pd.concat(all_results, ignore_index=True).to_csv(
            output_csv,
            sep=";", index=False, encoding="utf-8-sig",
        )

        print(
            f"  ИТОГО: строк={len(df_company)} | "
            f"переменных={df_company['Variable'].nunique()}"
        )
        print(f"  timings: {session.report_timings()}")

        if sleep_s:
            time.sleep(sleep_s)

    if not all_results:
        print("Нет результатов")
        return pd.DataFrame(columns=HEADER_COLS)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(output_csv, sep=";", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Готово:", output_csv)
    print("Компаний:", final["Company_name"].nunique())
    print("Строк:", len(final))
    print("Переменных на компанию:")
    print(final.groupby("Company_name")["Variable"].nunique())

    return final

# =====================================================================
# Диагностический режим retrieval-audit (RAG_DIAGNOSTIC=1)
#
# - НЕ вызывает LLM и gateway вообще (только чанкинг + лексика/эмбеддинги).
# - НЕ влияет на основной пайплайн: включается только через RAG_DIAGNOSTIC=1.
# - Ничего не пишет на диск VM: весь результат — stdout (консоль),
#   при запуске из esg_pipeline.ipynb stdout сохраняется локально.
# - Для каждой пары Company × Variable выводит таблицу:
#     level | document_id | chunk_id | score | rank | selected | reason
#   (+ positive_score, negative_penalty, rank_without_negative, url, title)
# - reasons: document level  -> "top document" / "below candidate_k" / "score<=0"
#             chunk level     -> "final" / "duplicate" / "diversity limit"
#                              / "below final_k" / "below candidate_k"
# =====================================================================
RAG_DIAG_DOC_K = int(os.environ.get("RAG_DIAG_DOC_K", "40"))
RAG_DIAG_CHUNK_CAND_K = int(os.environ.get("RAG_DIAG_CHUNK_CAND_K", "150"))
RAG_DIAG_MAX_CHUNKS_PER_DOC = int(os.environ.get("RAG_DIAG_MAX_CHUNKS_PER_DOC", "3"))
RAG_DIAG_VARIABLES = [v.strip() for v in os.environ.get("RAG_DIAG_VARIABLES", "").split(",") if v.strip()]
RAG_DIAG_FULL_TABLE = os.environ.get("RAG_DIAG_FULL_TABLE", "1") == "1"
RAG_DIAG_PRINT_CSV = os.environ.get("RAG_DIAG_PRINT_CSV", "0") == "1"  # дублировать каждую таблицу в CSV в stdout
RAG_DIAG_SCORE_COMPONENTS = (os.environ.get("RAG_DIAG_SCORE_COMPONENTS", "0") == "1")

AUDIT_COLS = [
    "company", "variable", "level",
    "document_id", "chunk_id",
    "score", "rank", "selected", "reason",
    "positive_score", "negative_penalty", "rank_without_negative",
    "url", "title",
]


def _audit_chunk_key(text):
    key_text = re.sub(r"\s+", " ", str(text)[:700].lower())
    return hashlib.md5(key_text.encode("utf-8", errors="ignore")).hexdigest()

def audit_document_pair(docs, doc_i_a, doc_i_b):
    """
    Безопасный аудит двух документов.
    Ничего не изменяет в docs, dedupe или retrieval.
    LLM не используется.
    """

    if not (0 <= doc_i_a < len(docs) and 0 <= doc_i_b < len(docs)):
        print(
            f"[DOC PAIR AUDIT] Некорректные индексы: "
            f"{doc_i_a}, {doc_i_b}; docs={len(docs)}"
        )
        return

    a = docs[doc_i_a]
    b = docs[doc_i_b]

    def _value(doc, *keys):
        for key in keys:
            value = doc.get(key)
            if value not in (None, "", "nan", "NaN"):
                return str(value)
        return ""

    def _norm_text(text):
        text = str(text or "").lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    text_a = _norm_text(a.get("text", ""))
    text_b = _norm_text(b.get("text", ""))

    # Простая оценка буквального совпадения.
    # Это НЕ semantic similarity и используется только для диагностики.
    if text_a and text_b:
        shorter = min(len(text_a), len(text_b))
        prefix_match = (
            sum(
                x == y
                for x, y in zip(text_a[:shorter], text_b[:shorter])
            ) / shorter
        )
    else:
        prefix_match = 0.0

    print("\n" + "=" * 80)
    print(
        f"DOCUMENT PAIR AUDIT: "
        f"doc_{doc_i_a} <-> doc_{doc_i_b}"
    )
    print("=" * 80)

    fields = [
        ("company_id", "company_id"),
        ("source_id", "source_id"),
        ("title A", "page_title"),
        ("title B", "page_title"),
        ("url A", "url"),
        ("url B", "url"),
        ("final_url A", "final_url"),
        ("final_url B", "final_url"),
        ("file_path A", "file_path"),
        ("file_path B", "file_path"),
        ("created A", "created"),
        ("created B", "created"),
        ("last_modified A", "last_modified"),
        ("last_modified B", "last_modified"),
        ("parsing_date A", "parsing_date"),
        ("date_confidence A", "date_confidence"),
        ("date_confidence B", "date_confidence"),
        ("text_chars A", "text_chars"),
        ("text_chars B", "text_chars"),
    ]

    for label, key in fields:
        if label.endswith(" A") or label in {
            "company_id",
            "source_id",
            "title A",
            "title B",
            "url A",
            "url B",
            "final_url A",
            "final_url B",
            "file_path A",
            "file_path B",
            "created A",
            "created B",
            "last_modified A",
            "last_modified B",
            "parsing_date A",
            "date_confidence A",
            "date_confidence B",
            "text_chars A",
            "text_chars B",
        }:
            pass

    print("\n--- DOCUMENT A ---")
    for key in [
        "company_id",
        "source_id",
        "page_title",
        "url",
        "final_url",
        "file_path",
        "created",
        "last_modified",
        "parsing_date",
        "date_confidence",
        "text_chars",
    ]:
        print(f"{key}: {_value(a, key)}")

    print("\n--- DOCUMENT B ---")
    for key in [
        "company_id",
        "source_id",
        "page_title",
        "url",
        "final_url",
        "file_path",
        "created",
        "last_modified",
        "parsing_date",
        "date_confidence",
        "text_chars",
    ]:
        print(f"{key}: {_value(b, key)}")

    print("\n--- TEXT COMPARISON ---")
    print(f"text length A: {len(text_a)}")
    print(f"text length B: {len(text_b)}")
    print(f"length ratio: "
          f"{min(len(text_a), len(text_b)) / max(len(text_a), len(text_b), 1):.4f}")
    print(f"prefix character match: {prefix_match:.4f}")

    print("\n--- FIRST 1000 CHARACTERS ---")
    print("[A]")
    print(text_a[:1000])
    print("\n[B]")
    print(text_b[:1000])

    print("\n--- LAST 1000 CHARACTERS ---")
    print("[A]")
    print(text_a[-1000:])
    print("\n[B]")
    print(text_b[-1000:])

    print("=" * 80)

def audit_document_duplicates(
    session,
    docs,
    *,
    similarity_threshold=0.93,
    max_pairs=100,
    text_prefix_chars=3000,
):
    """
    Диагностика похожих документов после dedupe_docs.

    Ничего не изменяет в session/docs.
    LLM не используется.

    Проверяет:
      - cosine similarity embeddings документов;
      - URL;
      - название;
      - дату;
      - длину текста;
      - совпадение начала текста.

    Важно:
      session._embeddings должны соответствовать docs.
    """

    print("\n" + "=" * 100)
    print("DOCUMENT DUPLICATE AUDIT")
    print("=" * 100)

    n_docs = len(docs)

    print(f"Документов: {n_docs}")
    print(f"Порог cosine similarity: {similarity_threshold}")

    # ------------------------------------------------------------
    # 1. Получаем embeddings из session
    # ------------------------------------------------------------
    embeddings = getattr(session, "_embeddings", None)

    if embeddings is None:
        print("\n[ERROR] session._embeddings отсутствуют.")
        print("=" * 100)
        return

    embeddings = np.asarray(embeddings)

    print(f"Размер embeddings: {embeddings.shape}")

    if len(embeddings) != n_docs:
        print(
            f"\n[ERROR] Количество embeddings ({len(embeddings)}) "
            f"не совпадает с количеством docs ({n_docs})."
        )
        print("=" * 100)
        return

    # ------------------------------------------------------------
    # 2. Нормализация embeddings
    # ------------------------------------------------------------
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    emb_norm = embeddings / norms

    # ------------------------------------------------------------
    # 3. Матрица cosine similarity
    # ------------------------------------------------------------
    similarity_matrix = emb_norm @ emb_norm.T

    # ------------------------------------------------------------
    # 4. Собираем пары
    # ------------------------------------------------------------
    pairs = []

    for i in range(n_docs):
        for j in range(i + 1, n_docs):

            sim = float(similarity_matrix[i, j])

            if sim >= similarity_threshold:
                pairs.append((sim, i, j))

    pairs.sort(reverse=True)

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    print(f"Найдено пар выше порога: {len(pairs)}")

    if not pairs:
        print("\nПохожих документов не найдено.")
        print("=" * 100)
        return

    # ------------------------------------------------------------
    # 5. Вспомогательная нормализация текста
    # ------------------------------------------------------------
    def normalize_text(text):
        text = str(text or "").lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------
    # 6. Вывод
    # ------------------------------------------------------------
    for pair_no, (sim, i, j) in enumerate(pairs, 1):

        doc_a = docs[i]
        doc_b = docs[j]

        text_a = normalize_text(doc_a.get("text", ""))
        text_b = normalize_text(doc_b.get("text", ""))

        prefix_a = text_a[:text_prefix_chars]
        prefix_b = text_b[:text_prefix_chars]

        if prefix_a == prefix_b and prefix_a:
            text_status = "IDENTICAL_PREFIX"

        elif prefix_a[:1000] == prefix_b[:1000] and prefix_a:
            text_status = "VERY_SIMILAR_PREFIX"

        else:
            text_status = "DIFFERENT_PREFIX"

        print("\n" + "-" * 100)
        print(f"PAIR #{pair_no}")
        print(f"cosine_similarity = {sim:.4f}")
        print(f"text_status       = {text_status}")

        print("\nDOCUMENT A")
        print(f"  id          = doc_{i}")
        print(f"  source_id   = {doc_a.get('source_id', '')}")
        print(f"  title       = {str(doc_a.get('page_title', ''))[:200]}")
        print(f"  url         = {str(doc_a.get('url', ''))[:300]}")
        print(f"  final_url   = {str(doc_a.get('final_url', ''))[:300]}")
        print(f"  created     = {doc_a.get('created', '')}")
        print(f"  modified    = {doc_a.get('last_modified', '')}")
        print(f"  text_chars  = {doc_a.get('text_chars', len(text_a))}")

        print("\nDOCUMENT B")
        print(f"  id          = doc_{j}")
        print(f"  source_id   = {doc_b.get('source_id', '')}")
        print(f"  title       = {str(doc_b.get('page_title', ''))[:200]}")
        print(f"  url         = {str(doc_b.get('url', ''))[:300]}")
        print(f"  final_url   = {str(doc_b.get('final_url', ''))[:300]}")
        print(f"  created     = {doc_b.get('created', '')}")
        print(f"  modified    = {doc_b.get('last_modified', '')}")
        print(f"  text_chars  = {doc_b.get('text_chars', len(text_b))}")

        if text_status in ("IDENTICAL_PREFIX", "VERY_SIMILAR_PREFIX"):

            print("\nTEXT A (first 1000 chars)")
            print(prefix_a[:1000])

            print("\nTEXT B (first 1000 chars)")
            print(prefix_b[:1000])

    print("\n" + "=" * 100)
    print("DOCUMENT DUPLICATE AUDIT FINISHED")
    print("=" * 100)

def audit_variable_retrieval(session, chunks, var, *,
                             document_k=RAG_DIAG_DOC_K,
                             chunk_cand_k=RAG_DIAG_CHUNK_CAND_K,
                             final_k=20,
                             max_per_doc=RAG_DIAG_MAX_CHUNKS_PER_DOC):
    """Детерминированный audit одной переменной одной компании. Без LLM."""
    scorer = session.get_scorer()
    scores_full = session.get_scores(var)                       # как в проде: lexical(+negative) + semantic
    scores_lex = np.asarray(scorer.score(var), dtype=np.float64)          # lexical с negative
    scores_pos = np.asarray(scorer.score(var, include_negative=False),
                            dtype=np.float64)                  # lexical без negative

    rows = []

    # ---------- DOCUMENT LEVEL: score документа = max score его чанков ----------
    doc_best = {}
    doc_url = {}
    doc_title = {}
    for i, ch in enumerate(chunks):
        di = ch["doc_i"]
        s = float(scores_full[i])
        if di not in doc_best or s > doc_best[di]:
            doc_best[di] = s
        doc_url[di] = ch.get("url", "")
        doc_title[di] = ch.get("title", "")

    ranked_docs = sorted(doc_best.items(), key=lambda kv: -kv[1])
    selected_docs = set()
    for rank, (di, s) in enumerate(ranked_docs, 1):
        is_sel = rank <= document_k and s > 0
        if is_sel:
            selected_docs.add(di)
            reason = "top document"
        elif s <= 0:
            reason = "score<=0"
        else:
            reason = "below candidate_k"
        rows.append({
            "level": "document",
            "document_id": f"doc_{di}",
            "chunk_id": "",
            "score": round(s, 3),
            "rank": rank,
            "selected": "yes" if is_sel else "no",
            "reason": reason,
            "positive_score": "", "negative_penalty": "", "rank_without_negative": "",
            "url": doc_url.get(di, ""), "title": doc_title.get(di, ""),
        })

    # ---------- CHUNK LEVEL: только чанки документов из selected_docs ----------
    cand = [
        (i, float(scores_full[i]))
        for i, ch in enumerate(chunks)
        if ch["doc_i"] in selected_docs and float(scores_full[i]) > 0
    ]
    cand.sort(key=lambda x: -x[1])

    # ------------------------------------------------------------
    # ПОЛНЫЙ RANKING ЧАНКОВ
    # ------------------------------------------------------------
    # cand содержит ВСЕ положительно оцененные чанки
    # выбранных document-level кандидатов.
    #
    # Здесь сохраняем их полный ранг, включая позиции > chunk_cand_k.
    # Это только диагностика и никак не меняет production retrieval.
    all_chunk_rank = {
        i: r
        for r, (i, _) in enumerate(cand, 1)
    }

    # A/B negative:
    # ранг того же чанка по positive-only score
    pos_sorted = sorted(
        cand,
        key=lambda x: -float(scores_pos[x[0]])
    )

    rank_wo_neg = {
        i: r
        for r, (i, _) in enumerate(pos_sorted, 1)
    }

    # ------------------------------------------------------------
    # PRODUCTION-LIKE CANDIDATE POOL
    # ------------------------------------------------------------
    pool = cand[:chunk_cand_k]

    pool_rank = {
        i: r
        for r, (i, _) in enumerate(pool, 1)
    }

    pool_set = {i for i, _ in pool}
        # ------------------------------------------------------------
    # ДИАГНОСТИКА РАСПРЕДЕЛЕНИЯ РАНГОВ
    # ------------------------------------------------------------

    rank_blocks = [
        ("1-20", 1, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-150", 101, 150),
        ("151+", 151, float("inf")),
    ]

    print(f"\n    CHUNK RANK DISTRIBUTION: {var}")

    for block_name, lo, hi in rank_blocks:
        block = [
            (i, s)
            for i, s in cand
            if lo <= all_chunk_rank[i] <= hi
        ]

        if not block:
            print(
                f"      {block_name}: "
                f"0 chunks"
            )
            continue

        unique_docs = len({
            chunks[i]["doc_i"]
            for i, _ in block
        })

        selected_final = sum(
            1 for i, _ in block
            if all_chunk_rank[i] <= final_k
        )

        print(
            f"      {block_name}: "
            f"chunks={len(block)}, "
            f"unique_docs={unique_docs}, "
            f"score_max={max(s for _, s in block):.3f}, "
            f"score_min={min(s for _, s in block):.3f}"
        )

        # ------------------------------------------------------------
    # ПОДРОБНЫЙ ВЫВОД КАНДИДАТОВ ЗА TOP-20
    # ------------------------------------------------------------

    print(f"\n    CHUNKS BELOW FINAL_K: {var}")

    for i, s in cand:
        rank = all_chunk_rank[i]

        if rank <= final_k:
            continue

        ch = chunks[i]

        print(
            f"      rank={rank:>4} | "
            f"doc=doc_{ch['doc_i']} | "
            f"chunk={ch.get('chunk_id', f'{ch['doc_i']}_{ch.get('chunk_i', '')}')} | "
            f"score={s:.3f} | "
            f"positive={float(scores_pos[i]):.3f} | "
            f"negative={float(scores_lex[i] - scores_pos[i]):.3f} | "
            f"rank_wo_neg={rank_wo_neg.get(i, '')} | "
            f"title={str(ch.get('title', ''))[:120]} | "
            f"url={str(ch.get('url', ''))[:160]}"
        )
    pool_rank = {i: r for r, (i, _) in enumerate(pool, 1)}
    pool_set = {i for i, _ in pool}

    seen_keys = set()
    final_count = 0
    per_doc = {}

    # проходим pool в порядке rank: dedup -> diversity -> final_k
    for i, s in pool:
        ch = chunks[i]
        key = _audit_chunk_key(ch["text"])
        if key in seen_keys:
            selected, reason = "no", "duplicate"
        elif final_count >= final_k:
            selected, reason = "no", "below final_k"
        elif per_doc.get(ch["doc_i"], 0) >= max_per_doc:
            selected, reason = "no", "diversity limit"
        else:
            seen_keys.add(key)
            per_doc[ch["doc_i"]] = per_doc.get(ch["doc_i"], 0) + 1
            final_count += 1
            selected, reason = "yes", "final"
        rows.append({
            "level": "chunk",
            "document_id": f"doc_{ch['doc_i']}",
            "chunk_id": ch.get("chunk_id", f"{ch['doc_i']}_{ch.get('chunk_i', '')}"),
            "score": round(s, 3),
            # "rank": pool_rank[i],
            "rank": all_chunk_rank[i],
            "selected": selected,
            "reason": reason,
            "positive_score": round(float(scores_pos[i]), 3),
            "negative_penalty": round(float(scores_lex[i] - scores_pos[i]), 3),
            "rank_without_negative": rank_wo_neg.get(i, ""),
            "url": ch.get("url", ""), "title": ch.get("title", ""),
        })

    # чанки кандидатных документов, не попавшие в candidate pool
    n_below = 0
    for i, ch in enumerate(chunks):
        if ch["doc_i"] in selected_docs and float(scores_full[i]) > 0 and i not in pool_set:
            n_below += 1
    if n_below:
        rows.append({
            "level": "chunk", "document_id": "", "chunk_id": "",
            "score": "", "rank": "", "selected": "no",
            "reason": f"below candidate_k (еще {n_below} чанков кандидатных документов вне топ-{chunk_cand_k})",
            "positive_score": "", "negative_penalty": "", "rank_without_negative": "",
            "url": "", "title": "",
        })

    return rows

def diagnose_score_composition(
    session,
    chunks,
    var,
    stage_queries=None,
    top_k=20,
    max_matches_per_chunk=12,
    context_chars=180,
):
    """
    Диагностика причин высокого score для конкретной ESG-переменной.

    Функция:
    - НЕ изменяет session;
    - НЕ изменяет chunks;
    - НЕ запускает LLM;
    - использует уже существующий LexicalScorer;
    - отдельно показывает lexical и semantic составляющие score.
    """

    print("\n" + "=" * 100)
    print(f"[SCORE DIAGNOSTIC] variable={var}")
    print("=" * 100)

    # ------------------------------------------------------------
    # 2. Lexical scores
    # ------------------------------------------------------------
    scorer = session.get_scorer()

    scores_lex = scorer.score(
        var,
        stage_queries,
        include_negative=True,
    )

    scores_pos = scorer.score(
        var,
        stage_queries,
        include_negative=False,
    )

    # ------------------------------------------------------------
    # 3. Semantic scores
    # ------------------------------------------------------------
    semantic_scores = np.zeros(len(chunks), dtype=float)

    if (
        getattr(session, "enable_semantic", False)
        and getattr(session, "_embeddings", None) is not None
    ):
        q = encode_stage_query(var, stage_queries)

        semantic_scores = session._embeddings @ q
        semantic_scores = np.asarray(semantic_scores, dtype=float)

    semantic_weight = float(
        getattr(session, "semantic_weight", 0.0)
    )

    semantic_contribution = semantic_weight * semantic_scores

    # ------------------------------------------------------------
    # 4. Итоговый score
    # ------------------------------------------------------------
    final_scores = (
        np.asarray(scores_lex, dtype=float)
        + semantic_contribution
    )

    order = np.argsort(-final_scores, kind="stable")

    # ------------------------------------------------------------
    # 5. Получаем слова/фразы, используемые scorer
    # ------------------------------------------------------------
    positive_terms = []
    negative_terms = []

    # Пытаемся взять уже подготовленные термы из scorer.
    # Если структура scorer отличается, функция не падает.
    for attr_name in (
        "positive_terms",
        "_positive_terms",
        "terms",
        "_terms",
    ):
        value = getattr(scorer, attr_name, None)

        if isinstance(value, (list, tuple, set)):
            positive_terms = list(value)
            break

        if isinstance(value, dict):
            positive_terms = list(value.keys())
            break

    for attr_name in (
        "negative_terms",
        "_negative_terms",
    ):
        value = getattr(scorer, attr_name, None)

        if isinstance(value, (list, tuple, set)):
            negative_terms = list(value)
            break

        if isinstance(value, dict):
            negative_terms = list(value.keys())
            break

    # ------------------------------------------------------------
    # 6. Вспомогательная функция поиска совпадений
    # ------------------------------------------------------------
    def find_term_matches(text, terms):
        """
        Возвращает список:
        {
            term: ...,
            count: ...,
            contexts: [...]
        }
        """
        text = str(text or "")
        low = text.lower()

        results = []

        for term in terms:
            term = str(term).strip()

            if not term:
                continue

            term_low = term.lower()

            positions = []
            start = 0

            while True:
                pos = low.find(term_low, start)

                if pos < 0:
                    break

                positions.append(pos)
                start = pos + max(len(term_low), 1)

            if not positions:
                continue

            contexts = []

            for pos in positions[:max_matches_per_chunk]:
                left = max(0, pos - context_chars)
                right = min(
                    len(text),
                    pos + len(term) + context_chars
                )

                fragment = text[left:right]

                # Чтобы совпадение было заметнее
                contexts.append(
                    fragment.replace("\n", " ")
                )

            results.append({
                "term": term,
                "count": len(positions),
                "contexts": contexts,
            })

        results.sort(
            key=lambda x: (-x["count"], x["term"].lower())
        )

        return results

    # ------------------------------------------------------------
    # 7. Выводим stage queries
    # ------------------------------------------------------------
    print("\n[QUERY]")

    if stage_queries:
        if isinstance(stage_queries, (list, tuple)):
            for i, q in enumerate(stage_queries, 1):
                print(f"  {i}. {q}")
        else:
            print(f"  {stage_queries}")
    else:
        print("  stage_queries не найдены")

    # ------------------------------------------------------------
    # 8. Анализируем top-k
    # ------------------------------------------------------------
    print("\n" + "-" * 100)
    print(f"TOP {top_k} CHUNKS")
    print("-" * 100)

    for rank, idx in enumerate(order[:top_k], 1):

        idx = int(idx)
        ch = chunks[idx]

        lexical = float(scores_lex[idx])
        positive = float(scores_pos[idx])
        negative = lexical - positive

        semantic = float(semantic_scores[idx])
        semantic_part = float(semantic_contribution[idx])
        final = float(final_scores[idx])

        print("\n" + "=" * 100)
        print(
            f"RANK {rank} | "
            f"chunk={ch.get('chunk_id')} | "
            f"document={ch.get('document_id')}"
        )
        print("=" * 100)

        print(f"TITLE: {ch.get('title', '')}")
        print(f"URL:   {ch.get('url', '')}")

        print("\n[SCORES]")
        print(f"  positive lexical : {positive:.4f}")
        print(f"  negative penalty : {negative:.4f}")
        print(f"  lexical total    : {lexical:.4f}")
        print(f"  semantic cosine   : {semantic:.4f}")
        print(f"  semantic weighted : {semantic_part:.4f}")
        print(f"  FINAL SCORE       : {final:.4f}")

        # --------------------------------------------------------
        # Совпадения положительных терминов
        # --------------------------------------------------------
        text = str(ch.get("text", ""))

        pos_matches = find_term_matches(
            text,
            positive_terms,
        )

        neg_matches = find_term_matches(
            text,
            negative_terms,
        )

        print("\n[POSITIVE TERM MATCHES]")

        if not pos_matches:
            print("  Явных совпадений с извлечёнными термами не найдено.")
        else:
            for item in pos_matches:
                print(
                    f"  • {item['term']!r} "
                    f"(count={item['count']})"
                )

                for context in item["contexts"][:3]:
                    print(f"      ... {context} ...")

        print("\n[NEGATIVE TERM MATCHES]")

        if not neg_matches:
            print("  Совпадений не найдено.")
        else:
            for item in neg_matches:
                print(
                    f"  • {item['term']!r} "
                    f"(count={item['count']})"
                )

                for context in item["contexts"][:3]:
                    print(f"      ... {context} ...")

        # --------------------------------------------------------
        # Краткий фрагмент чанка
        # --------------------------------------------------------
        print("\n[CHUNK PREVIEW]")
        preview = re.sub(r"\s+", " ", text)

        if len(preview) > 700:
            preview = preview[:700] + "..."

        print(preview)

def diagnose_retrieval_components(
    session,
    chunks,
    var,
    stage_queries=None,
    top_k=50,
    max_matches_per_chunk=12,
):
    """
    Диагностика состава retrieval-score.

    Ничего не меняет в production retrieval:
    - не вызывает LLM;
    - не меняет веса;
    - не меняет chunks;
    - не меняет результаты retrieve_and_rerank;
    - не записывает ничего на VM/S3.

    Сравниваются:
      production
      lexical_full
      lexical_no_blob
      query_only
      semantic_only
    """

    scorer = session.get_scorer()

    queries = (
        stage_queries
        if stage_queries is not None
        else ESG_QUERIES.get(var, [])
    )

    terms = tokenize_query_terms(queries)
    k = max(1, len(terms))

    # ------------------------------------------------------------
    # 1. Production components
    # ------------------------------------------------------------

    lexical_full = scorer.score(
        var,
        stage_queries,
        include_negative=True,
    )

    lexical_positive = scorer.score(
        var,
        stage_queries,
        include_negative=False,
    )

    negative_penalty = lexical_full - lexical_positive

    # ------------------------------------------------------------
    # 2. Semantic component
    # ------------------------------------------------------------

    semantic_weighted = np.zeros(len(chunks), dtype=float)
    semantic_cosine = np.zeros(len(chunks), dtype=float)

    if (
        getattr(session, "enable_semantic", False)
        and getattr(session, "_embeddings", None) is not None
    ):
        q = encode_stage_query(var, stage_queries)

        embeddings = np.asarray(session._embeddings)

        if len(embeddings) == len(chunks):
            semantic_cosine = embeddings @ q
            semantic_weighted = (
                float(session.semantic_weight) * semantic_cosine
            )

    production_score = lexical_full + semantic_weighted

    # ------------------------------------------------------------
    # 3. Exact decomposition of lexical score
    # ------------------------------------------------------------

    vocab_index = scorer.vocab_index

    ids = [
        vocab_index[t]
        for t in terms
        if t in vocab_index
    ]

    if ids:
        id_arr = np.asarray(ids, dtype=int)

        hits_body = (
            scorer.body_pres[:, id_arr]
            .sum(axis=1)
            .astype(float)
        )

        hits_blob = (
            scorer.blob_pres[:, id_arr]
            .sum(axis=1)
            .astype(float)
        )
    else:
        hits_body = np.zeros(len(chunks), dtype=float)
        hits_blob = np.zeros(len(chunks), dtype=float)

    body_term_score = (
        hits_body * (1.4 + 8.0 / k)
    )

    metadata_term_score = hits_blob * 1.1

    # ------------------------------------------------------------
    # 4. Phrase matches
    # ------------------------------------------------------------

    phrase_body_score = np.zeros(len(chunks), dtype=float)
    phrase_metadata_score = np.zeros(len(chunks), dtype=float)

    phrase_matches_body = [[] for _ in chunks]
    phrase_matches_blob = [[] for _ in chunks]

    for ph in queries:
        p = _norm_text_for_score(ph)

        if not p:
            continue

        if p not in scorer._phrase_cache:
            scorer._phrase_cache[p] = (
                np.fromiter(
                    (
                        p in low
                        for low in scorer.lows
                    ),
                    dtype=np.float64,
                    count=scorer.n,
                ),
                np.fromiter(
                    (
                        p in blob
                        for blob in scorer.blobs
                    ),
                    dtype=np.float64,
                    count=scorer.n,
                ),
            )

        pb, pbl = scorer._phrase_cache[p]

        phrase_body_score += 5.0 * pb
        phrase_metadata_score += 3.0 * pbl

        for i in np.flatnonzero(pb):
            if len(phrase_matches_body[i]) < max_matches_per_chunk:
                phrase_matches_body[i].append(ph)

        for i in np.flatnonzero(pbl):
            if len(phrase_matches_blob[i]) < max_matches_per_chunk:
                phrase_matches_blob[i].append(ph)

    # ------------------------------------------------------------
    # 5. Query-only score
    #
    # Здесь специально НЕТ:
    #   const
    #   evidence_score
    #   priority_score
    #   regex_score
    #   negative penalties
    #
    # Но metadata-term и metadata-phrase оставляем отдельно,
    # чтобы видеть их вклад.
    # ------------------------------------------------------------

    query_only_body = (
        body_term_score
        + phrase_body_score
    )

    query_only_with_metadata = (
        query_only_body
        + metadata_term_score
        + phrase_metadata_score
    )

    # ------------------------------------------------------------
    # 6. Lexical без metadata/blob
    # ------------------------------------------------------------

    lexical_no_blob = (
        scorer.const
        + scorer.evidence_score
        + scorer.regex_score
        + body_term_score
        + phrase_body_score
        + negative_penalty
    )

    # ------------------------------------------------------------
    # 7. Проверка точного разложения
    # ------------------------------------------------------------

    lexical_reconstructed = (
        scorer.const
        + scorer.evidence_score
        + scorer.priority_score
        + scorer.regex_score
        + body_term_score
        + metadata_term_score
        + phrase_body_score
        + phrase_metadata_score
        + negative_penalty
    )

    reconstruction_error = (
        lexical_full - lexical_reconstructed
    )

    # ------------------------------------------------------------
    # 8. Ранжирование
    # ------------------------------------------------------------

    order = np.argsort(
        -production_score,
        kind="stable",
    )

    top_indices = order[:top_k]

    # ------------------------------------------------------------
    # 9. Заголовок диагностики
    # ------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        f"RETRIEVAL COMPONENTS | {session.company_id} | {var}"
    )
    print("=" * 100)

    print(f"queries: {queries}")
    print(f"terms: {terms}")
    print(f"k terms: {k}")
    print(f"semantic enabled: {session.enable_semantic}")
    print(f"semantic weight: {session.semantic_weight}")

    print()
    print(
        "score reconstruction max error: "
        f"{np.max(np.abs(reconstruction_error)):.10f}"
    )

    # ------------------------------------------------------------
    # 10. Сводная статистика
    # ------------------------------------------------------------

    def _mean_top(arr):
        return float(
            np.mean(arr[top_indices])
        ) if len(top_indices) else 0.0

    print()
    print("TOP-K MEANS")
    print(
        f"production       = {_mean_top(production_score):.4f}"
    )
    print(
        f"lexical_full     = {_mean_top(lexical_full):.4f}"
    )
    print(
        f"semantic         = {_mean_top(semantic_weighted):.4f}"
    )
    print(
        f"lexical_no_blob  = {_mean_top(lexical_no_blob):.4f}"
    )
    print(
        f"query_only_body  = {_mean_top(query_only_body):.4f}"
    )
    print(
        f"query_only+meta  = {_mean_top(query_only_with_metadata):.4f}"
    )
    print(
        f"const             = {_mean_top(scorer.const):.4f}"
    )
    print(
        f"evidence          = {_mean_top(scorer.evidence_score):.4f}"
    )
    print(
        f"priority          = {_mean_top(scorer.priority_score):.4f}"
    )
    print(
        f"regex             = {_mean_top(scorer.regex_score):.4f}"
    )
    print(
        f"body_terms        = {_mean_top(body_term_score):.4f}"
    )
    print(
        f"metadata_terms    = {_mean_top(metadata_term_score):.4f}"
    )
    print(
        f"phrase_body       = {_mean_top(phrase_body_score):.4f}"
    )
    print(
        f"phrase_metadata   = {_mean_top(phrase_metadata_score):.4f}"
    )
    print(
        f"negative_penalty  = {_mean_top(negative_penalty):.4f}"
    )

    # ------------------------------------------------------------
    # 11. Детальная таблица TOP-K
    # ------------------------------------------------------------

    header = (
        "rank | chunk_id | doc_id | "
        "prod | lex | sem | "
        "query | q+meta | "
        "const | evid | priority | regex | "
        "body | meta | ph_body | ph_meta | neg | "
        "body_hits | meta_hits | title"
    )

    print()
    print(header)
    print("-" * len(header))

    for rank, idx in enumerate(top_indices, start=1):

        idx = int(idx)
        ch = chunks[idx]

        doc_id = (
            ch.get("document_id")
            or ch.get("doc_id")
            or ""
        )

        chunk_id = (
            ch.get("chunk_id")
            or ""
        )

        title = str(
            ch.get("title")
            or ch.get("page_title")
            or ""
        ).replace("\n", " ")

        if len(title) > 70:
            title = title[:67] + "..."

        print(
            f"{rank:4d} | "
            f"{str(chunk_id):12s} | "
            f"{str(doc_id):16s} | "
            f"{production_score[idx]:6.2f} | "
            f"{lexical_full[idx]:6.2f} | "
            f"{semantic_weighted[idx]:6.2f} | "
            f"{query_only_body[idx]:6.2f} | "
            f"{query_only_with_metadata[idx]:6.2f} | "
            f"{scorer.const[idx]:5.2f} | "
            f"{scorer.evidence_score[idx]:5.2f} | "
            f"{scorer.priority_score[idx]:5.2f} | "
            f"{scorer.regex_score[idx]:5.2f} | "
            f"{body_term_score[idx]:5.2f} | "
            f"{metadata_term_score[idx]:5.2f} | "
            f"{phrase_body_score[idx]:7.2f} | "
            f"{phrase_metadata_score[idx]:7.2f} | "
            f"{negative_penalty[idx]:5.2f} | "
            f"{int(hits_body[idx]):9d} | "
            f"{int(hits_blob[idx]):9d} | "
            f"{title}"
        )

    # ------------------------------------------------------------
    # 12. Отдельно показываем проблемные случаи
    # ------------------------------------------------------------

    print()
    print("POTENTIAL PROBLEM CASES")
    print(
        "Сначала идут документы, у которых production высокий, "
        "но query-only заметно ниже."
    )

    gap = production_score - query_only_body

    problem_order = np.argsort(
        -gap,
        kind="stable",
    )[:min(top_k, len(chunks))]

    print()
    print(
        "rank | chunk_id | production | query_only | "
        "gap | evid | priority | regex | const | title"
    )
    print("-" * 130)

    for rank, idx in enumerate(problem_order, start=1):

        idx = int(idx)
        ch = chunks[idx]

        title = str(
            ch.get("title")
            or ch.get("page_title")
            or ""
        ).replace("\n", " ")

        if len(title) > 75:
            title = title[:72] + "..."

        print(
            f"{rank:4d} | "
            f"{str(ch.get('chunk_id', '')):12s} | "
            f"{production_score[idx]:10.2f} | "
            f"{query_only_body[idx]:10.2f} | "
            f"{gap[idx]:6.2f} | "
            f"{scorer.evidence_score[idx]:5.2f} | "
            f"{scorer.priority_score[idx]:8.2f} | "
            f"{scorer.regex_score[idx]:5.2f} | "
            f"{scorer.const[idx]:5.2f} | "
            f"{title}"
        )

    # ------------------------------------------------------------
    # 13. Точные query matches для TOP-K
    # ------------------------------------------------------------

    print()
    print("QUERY MATCHES FOR TOP-K")

    for rank, idx in enumerate(top_indices, start=1):

        idx = int(idx)

        print()
        print(
            f"[rank={rank}] "
            f"chunk_id={chunks[idx].get('chunk_id', '')} "
            f"production={production_score[idx]:.3f}"
        )

        print(
            f"  body terms: "
            f"{int(hits_body[idx])} / {len(terms)}"
        )

        print(
            f"  metadata terms: "
            f"{int(hits_blob[idx])} / {len(terms)}"
        )

        if phrase_matches_body[idx]:
            print(
                "  body phrases: "
                + "; ".join(phrase_matches_body[idx])
            )
        else:
            print("  body phrases: none")

        if phrase_matches_blob[idx]:
            print(
                "  metadata phrases: "
                + "; ".join(phrase_matches_blob[idx])
            )
        else:
            print("  metadata phrases: none")

    print()
    print(
        "END RETRIEVAL COMPONENTS"
    )

# ============================================================
# A/B DIAGNOSTIC:
# PRODUCTION vs QUERY-ONLY vs QUERY-ONLY + SEMANTIC
# ============================================================

def _diagnostic_query_only_scores(scorer, var, stage_queries=None):
    """
    Только прямые совпадения терминов запроса в BODY чанка.

    Не используются:
      - metadata/blob
      - evidence_score
      - priority_score
      - regex_score
      - const
      - phrase bonuses
      - negative penalties

    Формула:
        query_only = body_hits * (1.4 + 8.0 / k)
    """

    queries = stage_queries if stage_queries is not None else ESG_QUERIES.get(var, [])
    terms = tokenize_query_terms(queries)

    ids = [
        scorer.vocab_index[t]
        for t in terms
        if t in scorer.vocab_index
    ]

    if not ids:
        return np.zeros(scorer.n, dtype=np.float64)

    id_arr = np.asarray(ids, dtype=int)

    hits_body = (
        scorer.body_pres[:, id_arr]
        .sum(axis=1)
        .astype(np.float64)
    )

    k = max(1, len(terms))

    return hits_body * (1.4 + 8.0 / k)


def _diagnostic_query_only_semantic_scores(
    scorer,
    chunks,
    var,
    stage_queries,
    semantic_weight=9.0, #19
):
    """
    Query-only lexical + semantic similarity.

    В lexical части используется только прямое совпадение
    терминов запроса в body.

    В semantic части используется существующий embedding
    CompanyRAGSession, если он уже рассчитан.
    """

    query_scores = _diagnostic_query_only_scores(
        scorer,
        var,
        stage_queries,
    )

    # Семантическая часть будет добавлена снаружи,
    # потому что embeddings принадлежат session.
    return query_scores


def _diagnostic_rank_positions(scores):
    """
    Возвращает:
        rank_by_idx[idx] = ранг документа/чанка, начиная с 1
    """
    order = np.argsort(-scores, kind="stable")

    rank_by_idx = np.empty(len(scores), dtype=np.int32)
    rank_by_idx[order] = np.arange(1, len(scores) + 1)

    return rank_by_idx


def _diagnostic_top_rows(
    chunks,
    scores_by_mode,
    *,
    top_k=20,
):
    """
    Формирует таблицу top-k для каждого режима.
    """

    rows = []

    for mode, scores in scores_by_mode.items():

        order = np.argsort(-scores, kind="stable")[:top_k]

        for rank, idx in enumerate(order, start=1):

            ch = chunks[int(idx)]

            rows.append({
                "mode": mode,
                "rank": rank,
                "chunk_index": int(idx),
                "document_id": str(
                    ch.get("document_id")
                    or ch.get("doc_id")
                    or ch.get("id")
                    or ""
                ),
                "chunk_id": str(ch.get("chunk_id", "")),
                "score": float(scores[idx]),
                "title": str(ch.get("title", "")),
                "url": str(ch.get("url", "")),
                "text_preview": re.sub(
                    r"\s+",
                    " ",
                    str(ch.get("text", ""))[:500],
                ),
            })

    return rows


def _diagnostic_overlap(scores_a, scores_b, k):
    """
    Jaccard overlap top-k двух ранжирований.
    """

    order_a = np.argsort(-scores_a, kind="stable")[:k]
    order_b = np.argsort(-scores_b, kind="stable")[:k]

    set_a = set(int(x) for x in order_a)
    set_b = set(int(x) for x in order_b)

    union = set_a | set_b

    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


def _diagnostic_rank_change(
    scores_base,
    scores_new,
    *,
    top_k=20,
):
    """
    Показывает, какие чанки сильнее всего изменили положение
    при переходе от base к new.
    """

    rank_base = _diagnostic_rank_positions(scores_base)
    rank_new = _diagnostic_rank_positions(scores_new)

    order_new = np.argsort(-scores_new, kind="stable")[:top_k]

    rows = []

    for idx in order_new:

        idx = int(idx)

        rows.append({
            "chunk_index": idx,
            "rank_base": int(rank_base[idx]),
            "rank_new": int(rank_new[idx]),
            "rank_delta": int(
                rank_base[idx] - rank_new[idx]
            ),
            "score_base": float(scores_base[idx]),
            "score_new": float(scores_new[idx]),
        })

    return rows

def _diagnostic_lexical_ablation(
    scorer,
    var,
    stage_queries=None,
    top_k=20,
    chunks=None,
):
    """
    Диагностика влияния отдельных компонентов LexicalScorer.

    ВАЖНО:
    production-логика не изменяется.
    Компоненты временно обнуляются только на время
    диагностического вызова scorer.score().
    После каждого эксперимента исходные значения
    восстанавливаются.
    """
    if chunks is None:
        chunks = []

    def _top_indices(scores):
        scores = np.asarray(scores, dtype=float)

        order = np.argsort(
            -scores,
            kind="stable",
        )

        return [
            int(i)
            for i in order[:top_k]
            if float(scores[i]) > 0
        ]

    def _jaccard(a, b):
        a = set(a)
        b = set(b)

        union = a | b

        if not union:
            return 1.0

        return len(a & b) / len(union)

    # ------------------------------------------------------------
    # Production score
    # ------------------------------------------------------------

    production = np.asarray(
        scorer.score(
            var,
            stage_queries,
            include_negative=True,
        ),
        dtype=float,
    )

    # ------------------------------------------------------------
    # Вспомогательная функция для временного
    # отключения одного компонента
    # ------------------------------------------------------------

    def _score_without_component(component_name):
        original = getattr(
            scorer,
            component_name,
        )

        try:
            setattr(
                scorer,
                component_name,
                np.zeros_like(
                    np.asarray(original, dtype=float)
                ),
            )

            result = scorer.score(
                var,
                stage_queries,
                include_negative=True,
            )

            return np.asarray(
                result,
                dtype=float,
            )

        finally:
            setattr(
                scorer,
                component_name,
                original,
            )

    # ------------------------------------------------------------
    # Абляции
    # ------------------------------------------------------------

    modes = {
        "A_production": production,

        "B_no_evidence": _score_without_component(
            "evidence_score"
        ),

        "C_no_priority": _score_without_component(
            "priority_score"
        ),

        "D_no_regex": _score_without_component(
            "regex_score"
        ),

        "E_no_const": _score_without_component(
            "const"
        ),

        "F_no_negative": np.asarray(
            scorer.score(
                var,
                stage_queries,
                include_negative=False,
            ),
            dtype=float,
        ),
    }

    # ------------------------------------------------------------
    # Только query-зависимая часть.
    #
    # Убираем:
    #   const
    #   evidence
    #   priority
    #   regex
    #
    # Negative тоже выключаем.
    #
    # Поэтому здесь остаются body / metadata / phrase
    # компоненты, которые зависят от запроса.
    # ------------------------------------------------------------

    originals = {}

    for name in (
        "const",
        "evidence_score",
        "priority_score",
        "regex_score",
    ):
        originals[name] = getattr(scorer, name)

    try:
        for name in originals:
            value = originals[name]

            setattr(
                scorer,
                name,
                np.zeros_like(
                    np.asarray(value, dtype=float)
                ),
            )

        modes["G_query_dependent_only"] = np.asarray(
            scorer.score(
                var,
                stage_queries,
                include_negative=False,
            ),
            dtype=float,
        )

    finally:
        for name, value in originals.items():
            setattr(
                scorer,
                name,
                value,
            )

    # ------------------------------------------------------------
    # Top-k
    # ------------------------------------------------------------

    top_by_mode = {
        mode: _top_indices(scores)
        for mode, scores in modes.items()
    }

    # ------------------------------------------------------------
    # Jaccard относительно production
    # ------------------------------------------------------------

    production_top = top_by_mode["A_production"]

    print()
    print("=" * 100)
    print(f"LEXICAL COMPONENT ABLATION: {var}")
    print("=" * 100)

    print(
        f"{'mode':<28}"
        f"{'mean':>12}"
        f"{'median':>12}"
        f"{'max':>12}"
        f"{'Jaccard vs A':>16}"
    )

    print("-" * 100)

    jaccard = {}

    for mode, scores in modes.items():

        arr = np.asarray(
            scores,
            dtype=float,
        )

        jac = _jaccard(
            production_top,
            top_by_mode[mode],
        )

        jaccard[mode] = jac

        print(
            f"{mode:<28}"
            f"{np.mean(arr):>12.3f}"
            f"{np.median(arr):>12.3f}"
            f"{np.max(arr):>12.3f}"
            f"{jac:>16.3f}"
        )

    # ------------------------------------------------------------
    # Подробности по изменениям Top-k
    # ------------------------------------------------------------

    print()
    print("TOP-K CHANGES")
    print("-" * 100)

    for mode in modes:

        if mode == "A_production":
            continue

        base = set(
            top_by_mode["A_production"]
        )

        current = set(
            top_by_mode[mode]
        )

        removed = base - current
        added = current - base

        print()
        print(
            f"{mode}: "
            f"removed={len(removed)}, "
            f"added={len(added)}"
        )

        # --------------------------------------------------------
        # Новые чанки
        # --------------------------------------------------------

        if added:

            print("  ADDED:")

            ranked_added = sorted(
                added,
                key=lambda idx: (
                    -float(modes[mode][idx])
                ),
            )

            for idx in ranked_added[:10]:

                print(
                    f"    idx={idx} | "
                    f"score={modes[mode][idx]:.3f} | "
                    f"document_id="
                    f"{chunks[idx].get('document_id', '')} | "
                    f"chunk_id="
                    f"{chunks[idx].get('chunk_id', '')} | "
                    f"title="
                    f"{chunks[idx].get('title', '')[:120]}"
                )

        # --------------------------------------------------------
        # Выпавшие чанки
        # --------------------------------------------------------

        if removed:

            print("  REMOVED:")

            ranked_removed = sorted(
                removed,
                key=lambda idx: (
                    -float(production[idx])
                ),
            )

            for idx in ranked_removed[:10]:

                print(
                    f"    idx={idx} | "
                    f"production={production[idx]:.3f} | "
                    f"new={modes[mode][idx]:.3f} | "
                    f"document_id="
                    f"{chunks[idx].get('document_id', '')} | "
                    f"chunk_id="
                    f"{chunks[idx].get('chunk_id', '')} | "
                    f"title="
                    f"{chunks[idx].get('title', '')[:120]}"
                )

    return {
        "scores": modes,
        "top_by_mode": top_by_mode,
        "jaccard": jaccard,
    }

def _diagnostic_lexical_rank_table(
    chunks,
    scorer,
    var,
    ablation_result,
    extra_modes=None,
    top_k=20,
):
    """
    Детальное сравнение рангов одних и тех же чанков
    в production и диагностических режимах.

    Ничего не меняет в scorer и production-настройках.
    """

    extra_modes = extra_modes or {}

    # Все режимы из ablation
    modes = dict(ablation_result.get("scores", {}))

    # Дополнительные режимы, например semantic
    modes.update(extra_modes)

    if not modes:
        return None

    # --------------------------------------------------------
    # 1. Рассчитываем ранги внутри каждого режима
    # --------------------------------------------------------
    ranks_by_mode = {}
    top_indices_by_mode = {}

    for mode_name, scores in modes.items():
        scores = np.asarray(scores, dtype=float)

        order = np.argsort(-scores, kind="stable")

        ranks = np.full(len(scores), np.inf)
        ranks[order] = np.arange(1, len(scores) + 1)

        ranks_by_mode[mode_name] = ranks
        top_indices_by_mode[mode_name] = set(order[:top_k].tolist())

    # --------------------------------------------------------
    # 2. Берём объединение Top-K всех режимов
    # --------------------------------------------------------
    candidate_indices = set()

    for idx_set in top_indices_by_mode.values():
        candidate_indices.update(idx_set)

    # Сначала production rank, затем остальные
    production_ranks = ranks_by_mode.get("A_production")

    if production_ranks is not None:
        candidate_indices = sorted(
            candidate_indices,
            key=lambda i: (
                production_ranks[i],
                i,
            )
        )
    else:
        candidate_indices = sorted(candidate_indices)

    # --------------------------------------------------------
    # 3. Компоненты LexicalScorer
    # --------------------------------------------------------
    def _arr(name):
        value = getattr(scorer, name, None)

        if value is None:
            return np.zeros(len(chunks), dtype=float)

        arr = np.asarray(value, dtype=float)

        if len(arr) != len(chunks):
            return np.zeros(len(chunks), dtype=float)

        return arr

    evidence = _arr("evidence_score")
    priority = _arr("priority_score")
    regex = _arr("regex_score")
    body_term = _arr("body_term_score")
    metadata_term = _arr("metadata_term_score")

    # --------------------------------------------------------
    # 4. Печать таблицы
    # --------------------------------------------------------
    print("\n" + "=" * 150)
    print(f"DETAILED LEXICAL RANK COMPARISON: {var}")
    print(
        "Объединение Top-K чанков из production, ablation и semantic-режимов"
    )
    print("=" * 150)

    headers = [
        "chunk",
        "title",
        "prod_r",
        "no_ev_r",
        "no_pr_r",
        "no_rg_r",
        "query_sem_r",
        "prod",
        "query_sem",
        "evid",
        "prior",
        "regex",
        "body",
        "meta",
    ]

    print(
        f"{headers[0]:<12} "
        f"{headers[1]:<52} "
        f"{headers[2]:>7} "
        f"{headers[3]:>7} "
        f"{headers[4]:>7} "
        f"{headers[5]:>7} "
        f"{headers[6]:>10} "
        f"{headers[7]:>8} "
        f"{headers[8]:>10} "
        f"{headers[9]:>7} "
        f"{headers[10]:>7} "
        f"{headers[11]:>7} "
        f"{headers[12]:>7} "
        f"{headers[13]:>7}"
    )

    print("-" * 150)

    def _rank(mode, idx):
        ranks = ranks_by_mode.get(mode)

        if ranks is None:
            return None

        r = ranks[idx]

        if np.isinf(r):
            return None

        return int(r)

    def _fmt_rank(mode, idx):
        r = _rank(mode, idx)
        return "-" if r is None else str(r)

    def _fmt_score(mode, idx):
        scores = modes.get(mode)

        if scores is None:
            return "-"

        return f"{float(scores[idx]):.2f}"

    # --------------------------------------------------------
    # 5. Вывод строк
    # --------------------------------------------------------
    for idx in candidate_indices:
        ch = chunks[idx]

        chunk_id = str(
            ch.get("chunk_id")
            or ch.get("id")
            or idx
        )

        title = str(
            ch.get("title")
            or ch.get("page_title")
            or ""
        )

        title = re.sub(r"\s+", " ", title).strip()

        if len(title) > 50:
            title = title[:47] + "..."

        print(
            f"{chunk_id:<12.12} "
            f"{title:<52.52} "
            f"{_fmt_rank('A_production', idx):>7} "
            f"{_fmt_rank('B_no_evidence', idx):>7} "
            f"{_fmt_rank('C_no_priority', idx):>7} "
            f"{_fmt_rank('D_no_regex', idx):>7} "
            f"{_fmt_rank('C_query_only_semantic', idx):>10} "
            f"{_fmt_score('A_production', idx):>8} "
            f"{_fmt_score('C_query_only_semantic', idx):>10} "
            f"{evidence[idx]:>7.2f} "
            f"{priority[idx]:>7.2f} "
            f"{regex[idx]:>7.2f} "
            f"{body_term[idx]:>7.2f} "
            f"{metadata_term[idx]:>7.2f}"
        )

    print("=" * 150)

    # --------------------------------------------------------
    # 6. Отдельно показываем наиболее сильные расхождения
    # --------------------------------------------------------
    if "A_production" in ranks_by_mode and "C_query_only_semantic" in ranks_by_mode:

        prod_ranks = ranks_by_mode["A_production"]
        query_ranks = ranks_by_mode["C_query_only_semantic"]

        shifts = []

        for idx in candidate_indices:
            pr = prod_ranks[idx]
            qr = query_ranks[idx]

            if np.isinf(pr):
                pr = len(chunks) + 1

            if np.isinf(qr):
                qr = len(chunks) + 1

            shift = int(pr) - int(qr)

            shifts.append((shift, idx))

        # Самые сильные случаи:
        # production rank намного хуже query-sem
        shifts.sort(reverse=True)

        print("\n" + "=" * 120)
        print(
            f"CHUNKS: production rank >> query-semantic rank | {var}"
        )
        print("=" * 120)

        for shift, idx in shifts[:20]:
            if shift <= 0:
                break

            ch = chunks[idx]

            chunk_id = str(
                ch.get("chunk_id")
                or ch.get("id")
                or idx
            )

            title = str(
                ch.get("title")
                or ch.get("page_title")
                or ""
            )

            title = re.sub(r"\s+", " ", title).strip()

            if len(title) > 70:
                title = title[:67] + "..."

            print(
                f"{chunk_id:<12} "
                f"prod={_fmt_rank('A_production', idx):>5} "
                f"query_sem={_fmt_rank('C_query_only_semantic', idx):>5} "
                f"shift={shift:>5} "
                f"prod_score={_fmt_score('A_production', idx):>7} "
                f"query_score={_fmt_score('C_query_only_semantic', idx):>7} "
                f"evid={evidence[idx]:>5.2f} "
                f"prior={priority[idx]:>5.2f} "
                f"regex={regex[idx]:>5.2f} "
                f"title={title}"
            )

        print("=" * 120)

    return {
        "modes": modes,
        "ranks": ranks_by_mode,
        "candidate_indices": candidate_indices,
    }

def diagnose_scoring_ab(
    session,
    chunks,
    var,
    *,
    final_k=20,
    top_k=20,
    semantic_weight=None,
    print_top=True,
):
    """
    Контролируемый A/B-тест scoring.

    Режимы:

    A production
        LexicalScorer.score(...)
        + semantic

    B query_only
        Только прямые body-term matches

    C query_only_semantic
        query_only + semantic

    D lexical_only
        Полный текущий lexical score без semantic

    Ничего не меняет в session/scorer/chunks.
    LLM не вызывается.
    """

    if semantic_weight is None:
        semantic_weight = float(
            getattr(session, "semantic_weight", 9.0) #19/18
        )

    scorer = session.get_scorer()

    stage_queries = ESG_QUERIES.get(var, [])

    # --------------------------------------------------------
    # 1. Production lexical
    # --------------------------------------------------------

    lexical_production = scorer.score(
        var,
        stage_queries,
        include_negative=True,
    ).copy()

    # --------------------------------------------------------
    # 2. Query-only
    # --------------------------------------------------------

    query_only = _diagnostic_query_only_scores(
        scorer,
        var,
        stage_queries,
    )

    # --------------------------------------------------------
    # 3. Semantic
    # --------------------------------------------------------

    semantic = np.zeros(len(chunks), dtype=np.float64)

    if (
        getattr(session, "_embeddings", None) is not None
        and len(session._embeddings) == len(chunks)
    ):
        q = encode_stage_query(
            var,
            stage_queries,
        )

        semantic = (
            session._embeddings @ q
        ).astype(np.float64)

    query_only_semantic = (
        query_only
        + semantic_weight * semantic
    )

    # --------------------------------------------------------
    # 4. Production + semantic
    # --------------------------------------------------------

    production_semantic = (
        lexical_production
        + semantic_weight * semantic
    )

    # --------------------------------------------------------
    # 5. Проверка
    # --------------------------------------------------------

    scores_by_mode = {
        "A_production": lexical_production,
        "B_query_only": query_only,
        "C_query_only_semantic": query_only_semantic,
        "D_production_semantic": production_semantic,
    }
    lexical_ablation = _diagnostic_lexical_ablation(
        scorer=scorer,
        var=var,
        stage_queries=stage_queries,
        top_k=top_k,
        chunks=chunks,
    )
    lexical_rank_table = _diagnostic_lexical_rank_table(
        chunks=chunks,
        scorer=scorer,
        var=var,
        ablation_result=lexical_ablation,
        extra_modes={
            "C_query_only_semantic": query_only_semantic,
            "D_production_semantic": production_semantic,
        },
        top_k=top_k,
    )

    print()
    print("=" * 110)
    print(f"SCORING A/B DIAGNOSTIC | {session.company_id} | {var}")
    print("=" * 110)

    print(f"semantic enabled: {getattr(session, 'enable_semantic', False)}")
    print(f"semantic weight:  {semantic_weight:.4f}")
    print(f"chunks:           {len(chunks)}")
    print(f"final_k:          {final_k}")
    print()

    # --------------------------------------------------------
    # 6. Диапазоны score
    # --------------------------------------------------------

    print("SCORE RANGES")
    print("-" * 110)

    for mode, scores in scores_by_mode.items():

        print(
            f"{mode:25s} "
            f"min={np.min(scores):8.3f} "
            f"max={np.max(scores):8.3f} "
            f"mean={np.mean(scores):8.3f} "
            f"median={np.median(scores):8.3f}"
        )

    # --------------------------------------------------------
    # 7. Overlap top-k
    # --------------------------------------------------------

    print()
    print(f"TOP-{top_k} OVERLAP")
    print("-" * 110)

    mode_names = list(scores_by_mode.keys())

    for i in range(len(mode_names)):
        for j in range(i + 1, len(mode_names)):

            a = mode_names[i]
            b = mode_names[j]

            overlap = _diagnostic_overlap(
                scores_by_mode[a],
                scores_by_mode[b],
                top_k,
            )

            print(
                f"{a:25s} vs "
                f"{b:25s} "
                f"Jaccard={overlap:.3f}"
            )

    # --------------------------------------------------------
    # 8. TOP-K каждого режима
    # --------------------------------------------------------

    if print_top:

        for mode, scores in scores_by_mode.items():

            print()
            print("=" * 110)
            print(f"TOP-{top_k} | {mode}")
            print("=" * 110)

            order = np.argsort(
                -scores,
                kind="stable",
            )[:top_k]

            for rank, idx in enumerate(order, start=1):

                ch = chunks[int(idx)]

                title = str(
                    ch.get("title", "")
                ).replace("\n", " ")

                document_id = (
                    ch.get("document_id")
                    or ch.get("doc_id")
                    or ch.get("id")
                    or ""
                )

                chunk_id = ch.get(
                    "chunk_id",
                    "",
                )

                text_preview = re.sub(
                    r"\s+",
                    " ",
                    str(ch.get("text", ""))[:250],
                )

                print(
                    f"{rank:3d} | "
                    f"{str(document_id):15s} | "
                    f"{str(chunk_id):8s} | "
                    f"{float(scores[idx]):8.3f} | "
                    f"{title[:70]}"
                )

                print(
                    f"      {text_preview}"
                )

    # --------------------------------------------------------
    # 9. Кто поднялся/опустился
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        f"RANK CHANGES: A_production -> C_query_only_semantic"
    )
    print("=" * 110)

    rank_changes = _diagnostic_rank_change(
        lexical_production,
        query_only_semantic,
        top_k=top_k,
    )

    for row in rank_changes:

        idx = row["chunk_index"]
        ch = chunks[idx]

        document_id = (
            ch.get("document_id")
            or ch.get("doc_id")
            or ch.get("id")
            or ""
        )

        print(
            f"new_rank={row['rank_new']:3d} "
            f"old_rank={row['rank_base']:3d} "
            f"delta={row['rank_delta']:+4d} | "
            f"{str(document_id):15s} | "
            f"{str(ch.get('chunk_id', '')):8s} | "
            f"{str(ch.get('title', ''))[:70]}"
        )

    # --------------------------------------------------------
    # 10. Таблица для CSV
    # --------------------------------------------------------

    rows = _diagnostic_top_rows(
        chunks,
        scores_by_mode,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # 11. Возвращаем всё, но ничего не изменяем
    # --------------------------------------------------------

    return {
        "scores": scores_by_mode,
        "semantic": semantic,
        "query_only": query_only,
        "lexical_production": lexical_production,
        "rows": rows,
        "rank_changes": rank_changes,
    }

def run_retrieval_audit(meta_df=None, *,
                        variables=None,
                        document_k=RAG_DIAG_DOC_K,
                        chunk_cand_k=RAG_DIAG_CHUNK_CAND_K,
                        final_k=20,
                        max_per_doc=RAG_DIAG_MAX_CHUNKS_PER_DOC,
                        full_context_limit=80000,
                        max_chars_per_chunk=1600):
    """Прогон retrieval-audit по компаниям. Без LLM, без записи на диск."""
    if meta_df is None:
        meta_df = load_sources_index(SOURCES_DIR)

    company_ids = get_company_ids(meta_df)
    variables = list(variables) if variables else list(VARIABLES)

    print("=" * 80)
    print("RETRIEVAL AUDIT (диагностический режим, LLM НЕ используется)")
    print("=" * 80)
    print("Компаний к аудиту:", len(company_ids))
    print(f"params: doc_k={document_k}, chunk_cand_k={chunk_cand_k}, final_k={final_k}, "
          f"max_chunks_per_doc={max_per_doc}, semantic={'on' if RAG_ENABLE_SEMANTIC else 'off'}")
    print("=" * 80)

    for ci, company_id in enumerate(company_ids, 1):
        company_sources = meta_df[
            meta_df["company_id"].astype(str) == company_id
        ].to_dict("records")
        if not company_sources:
            continue

        print(f"\n[{ci}/{len(company_ids)}]  {company_id}")

        session = CompanyRAGSession(
            company_id,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        docs = session.get_docs(company_sources)
        # 
        audit_document_pair(docs, 788, 791)
        audit_document_pair(docs, 804, 805)
        # 
        if not docs:
            print("  пропуск: нет пригодных текстов")
            continue

        total_chars = sum(len(d["text"]) for d in docs)
        if total_chars <= full_context_limit:
            print(f"  [warn] {total_chars:,} <= full_context_limit={full_context_limit:,}: "
                  f"в проде retrieval пропущен и текст уходит в LLM целиком — "
                  f"аудит ниже показывает, ЧТО было бы выбрано при включенном retrieval")

        chunks = session.get_chunks(company_sources)
        print(f"  docs={len(docs)}, chunks={len(chunks)}")
                
        audit_document_duplicates(
            session,
            docs,
            similarity_threshold=0.93,
            max_pairs=100,
        )

        for var in variables:
            rows = audit_variable_retrieval(
                session, chunks, var,
                document_k=document_k,
                chunk_cand_k=chunk_cand_k,
                final_k=final_k,
                max_per_doc=max_per_doc,
            )

            if RAG_DIAG_SCORE_COMPONENTS:
                diagnose_retrieval_components(
                    session=session,
                    chunks=chunks,
                    var=var,
                    top_k=50,
                )

            diagnose_retrieval_components(
                session=session,
                chunks=chunks,
                var=var,
                top_k=20,
            )

            diagnose_score_composition(
                session=session,
                chunks=chunks,
                var=var,
                top_k=20,
            )

            for r in rows:
                r["company"] = company_id
                r["variable"] = var

            df_audit = pd.DataFrame(rows, columns=AUDIT_COLS)
            n_sel = int((df_audit["selected"] == "yes").sum())
            print(f"\n  --- {var} --- selected rows: {n_sel}")
            if RAG_DIAG_FULL_TABLE:
                print(df_audit[AUDIT_COLS[2:9]].to_string(index=False))
            if RAG_DIAG_PRINT_CSV:
                # машиночитаемый CSV прямо в stdout (для локального сохранения из ноутбука)
                print(df_audit.to_csv(sep=";", index=False))

        print(f"  timings: {session.report_timings()}")

    return None

def run_scoring_ab_diagnostic(
    company_id=None,
    variables=None,
    semantic=True,
    final_k=20,
    top_k=20,
    semantic_weight=None,
    save_local=True,
    local_dir="retrieval_ab",
):
    """
    Запускает только A/B-диагностику scoring.

    LLM не вызывается.
    Production-конфигурация не изменяется.
    Результат сохраняется только локально.
    """

    print()
    print("=" * 100)
    print("SCORING A/B DIAGNOSTIC")
    print("=" * 100)

    # --------------------------------------------------------
    # ВАЖНО:
    # здесь создаём отдельную session.
    # Никаких изменений production session.
    # --------------------------------------------------------

    if company_id is not None:
        company_ids = [company_id]
        meta_df = load_sources_index(SOURCES_DIR)
    else:
        meta_df = load_sources_index(SOURCES_DIR)
        company_ids = get_company_ids(meta_df)

    if variables is None:
        variables = list(ESG_QUERIES.keys())

    meta_df = load_sources_index(SOURCES_DIR)

    all_outputs = []

    for current_company_id in company_ids:

        print()
        print(
            f"COMPANY: {current_company_id}"
        )
        company_sources = meta_df[
            meta_df["company_id"].astype(str) == str(current_company_id)
        ].to_dict("records")

        if not company_sources:
            print("No company sources, skip.")
            continue

        session = CompanyRAGSession(
            current_company_id,
            enable_semantic=semantic,
            semantic_weight=(
                semantic_weight
                if semantic_weight is not None
                else RAG_SEMANTIC_WEIGHT
            ),
        )

        docs = session.get_docs(company_sources)

        if not docs:
            print("No documents, skip.")
            continue

        chunks = session.get_chunks(company_sources)

        if not chunks:
            print("No chunks, skip.")
            continue

        for var in variables:

            result = diagnose_scoring_ab(
                session,
                chunks,
                var,
                final_k=final_k,
                top_k=top_k,
                semantic_weight=semantic_weight,
                print_top=True,
            )

            all_outputs.append(
                (
                    current_company_id,
                    var,
                    result,
                )
            )

    return all_outputs

# if __name__ == "__main__":
#     print("Конфигурация загружена")
#     print("SOURCES_DIR:", SOURCES_DIR)
#     print("OUTPUT_CSV:", OUTPUT_CSV)
#     print("GATEWAY_URL:", GATEWAY_URL)
#     run_rag_by_variable(
#         output_csv=OUTPUT_CSV,
#         full_context_limit=int(os.environ.get("RAG_FULL_CONTEXT_LIMIT", "80000")),
#         recall_k=int(os.environ.get("RAG_RECALL_K", "48")),
#         final_k=int(os.environ.get("RAG_FINAL_K", "20")),
#         max_chars_per_chunk=int(os.environ.get("RAG_MAX_CHARS_PER_CHUNK", "1600")),
#         sleep_s=float(os.environ.get("RAG_SLEEP_S", "0.0")),
#     )

import sys

if __name__ == "__main__":

    print("Конфигурация загружена")
    print("SOURCES_DIR:", SOURCES_DIR)
    print("OUTPUT_CSV:", OUTPUT_CSV)
    print("GATEWAY_URL:", GATEWAY_URL)

    if os.environ.get("RAG_SCORING_AB_DIAGNOSTIC", "") == "1":

        run_scoring_ab_diagnostic(
            company_id=os.environ.get(
                "RAG_SCORING_AB_COMPANY_ID"
            ) or None,
            variables=(
                [
                    v.strip()
                    for v in os.environ.get(
                        "RAG_SCORING_AB_VARIABLES",
                        ""
                    ).split(",")
                    if v.strip()
                ]
                or None
            ),
            semantic=(
                os.environ.get(
                    "RAG_ENABLE_SEMANTIC",
                    "1"
                ) == "1"
            ),
            final_k=int(
                os.environ.get(
                    "RAG_FINAL_K",
                    "20"
                )
            ),
            top_k=int(
                os.environ.get(
                    "RAG_SCORING_AB_TOP_K",
                    "20"
                )
            ),
            semantic_weight=float(
                os.environ.get(
                    "RAG_SCORING_AB_SEMANTIC_WEIGHT",
                    "9.0"# 19/18
                )
            ),
        )

        sys.exit(0)

    elif os.environ.get("RAG_DIAGNOSTIC", "") == "1":

        run_retrieval_audit(
            variables=RAG_DIAG_VARIABLES or None,
            document_k=int(
                os.environ.get(
                    "RAG_DIAG_DOC_K",
                    "40"
                )
            ),
            chunk_cand_k=int(
                os.environ.get(
                    "RAG_DIAG_CHUNK_CAND_K",
                    "150"
                )
            ),
            final_k=int(
                os.environ.get(
                    "RAG_FINAL_K",
                    "20"
                )
            ),
            max_per_doc=int(
                os.environ.get(
                    "RAG_DIAG_MAX_CHUNKS_PER_DOC",
                    "3"
                )
            ),
            full_context_limit=int(
                os.environ.get(
                    "RAG_FULL_CONTEXT_LIMIT",
                    "80000"
                )
            ),
            max_chars_per_chunk=int(
                os.environ.get(
                    "RAG_MAX_CHARS_PER_CHUNK",
                    "1600"
                )
            ),
        )

        sys.exit(0)

    else:

        run_rag_by_variable(
            output_csv=OUTPUT_CSV,
            full_context_limit=int(
                os.environ.get(
                    "RAG_FULL_CONTEXT_LIMIT",
                    # "80000"
                    "90000"
                )
            ),
            recall_k=int(
                os.environ.get(
                    "RAG_RECALL_K",
                    "48"
                )
            ),
            final_k=int(
                os.environ.get(
                    "RAG_FINAL_K",
                    "20"
                )
            ),
            max_chars_per_chunk=int(
                os.environ.get(
                    "RAG_MAX_CHARS_PER_CHUNK",
                    "1600"
                )
            ),
            sleep_s=float(
                os.environ.get(
                    "RAG_SLEEP_S",
                    "0.0"
                )
            ),
        )