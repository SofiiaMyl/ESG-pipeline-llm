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

# =============================================================================
# НЕГАТИВНЫЕ ТЕРМИНЫ — против путаницы confusable-переменных
# =============================================================================
# Пары "переменная X часто путается с переменной Y" — структурная особенность
# ESG-таксономии (см. правила в STAGE2_SYSTEM_PROMPT), не зависит от отрасли.
# CONFUSABLE_PAIRS = {
#     "So4": ["So1"],
#     "So1": ["So4"],
#     "So2": ["So1"],
#     "So3": ["So1"],
#     "Ss3": ["Ss1"],
#     "Ss4": ["Ss1", "Ss2"],
#     "G2":  ["Sc1", "Sc2"],
#     "G3":  ["Sc4"],
#     "Se5": ["G3"],
# }

# def build_negative_queries():
#     """
#     Негативные термины для каждой переменной — собираются автоматически
#     из ESG_QUERIES её confusable-пар (CONFUSABLE_PAIRS), без ручного
#     подбора слов. Термины, совпадающие с собственным позитивным списком
#     переменной, исключаются, чтобы не наказывать саму себя.
#     """
#     neg = {}
#     for var, confused_with in CONFUSABLE_PAIRS.items():
#         terms = []
#         for other_var in confused_with:
#             terms.extend(ESG_QUERIES.get(other_var, []))
#         own_terms = {_norm_text_for_score(t) for t in ESG_QUERIES.get(var, [])}
#         neg[var] = [t for t in terms if _norm_text_for_score(t) not in own_terms]
#     return neg


# =============================================================================
# УНИВЕРСАЛЬНЫЕ НЕГАТИВНЫЕ ТЕРМИНЫ
# =============================================================================
# Используются только специфичные фразы, которые могут указывать,
# что фрагмент относится преимущественно к другой ESG-теме.
#
# Важно:
# - не использовать слишком общие слова вроде:
#   "community", "employee", "supplier", "training", "policy";
# - предпочтительно использовать устойчивые словосочетания;
# - словарь не зависит от отрасли компании.
#
# Формат:
#   "strong"  -> сильный штраф
#   "medium"  -> умеренный штраф
#   "weak"    -> пока не используется как штраф,
#                оставлен для дальнейших экспериментов.

# =============================================================================
# УНИВЕРСАЛЬНЫЕ НЕГАТИВНЫЕ ТЕРМИНЫ
# Русский + английский
#
# Используются не как жесткий фильтр, а только как мягкий штраф
# при лексическом reranking.
#
# Термины сгруппированы по смысловым категориям. Это позволяет:
# 1) не штрафовать несколько слов одной и той же категории отдельно;
# 2) использовать один словарь для компаний разных отраслей;
# 3) не зависеть от конкретного названия компании;
# 4) учитывать русскоязычные и англоязычные документы.
# =============================================================================

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
RAG_SEMANTIC_WEIGHT = float(os.environ.get("RAG_SEMANTIC_WEIGHT", "18.0"))

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
                self.evidence_score += 0.9 * mult * np.fromiter(
                    (marker in low for low in self.lows), dtype=np.float64, count=n
                )

        # --- priority-маркеры: по source_blob (тело[:4000] + title + url + file) ---
        pr_counts = _unique_with_multiplicity(
            _norm_text_for_score(x) for x in SOURCE_PRIORITY_MARKERS
        )
        self.priority_score = np.zeros(n)
        for marker, mult in pr_counts.items():
            if marker:
                self.priority_score += 1.0 * mult * np.fromiter(
                    (marker in hb for hb in self.head_blobs), dtype=np.float64, count=n
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

    def score(self, var, stage_queries=None):
        """Вектор лексических score для переменной, np.array shape (n_chunks,)."""
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

        if var in STAGE2_VARIABLES:
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

def dedupe_similar_docs(chunks, docs, threshold=RAG_DOC_DEDUPE_THRESHOLD):
    """
    Схлопывает документы-переводы одной страницы по средней эмбеддинг-близости
    их чанков. PDF-документы исключаются из дедупликации целиком — у них не
    бывает языковых копий, а структурное сходство (одинаковый шаблон, разные
    даты/цифры) не означает дублирование содержания.
    Не объединяет документы, если у них распознаны разные даты (за пределами
    допуска RAG_DOC_DEDUPE_DATE_TOLERANCE_DAYS).
    Оставляет один представитель на кластер (самый длинный текст).
    Для каждой схлопнутой пары выводит similarity и названия документов —
    для калибровки порога RAG_DOC_DEDUPE_THRESHOLD.
    """
    doc_dates = {}
    doc_lens = {}
    for i, d in enumerate(docs):
        doc_lens[i] = len(d.get("text", ""))
        doc_dates[i] = d.get("date") or d.get("created") or d.get("lastmodified") or d.get("last_modified") or ""

    by_doc = {}
    for ch in chunks:
        by_doc.setdefault(ch["doc_i"], []).append(ch["embedding"])

    by_doc_vecs = {}
    for doc_i, vecs in by_doc.items():
        v = np.mean(vecs, axis=0)
        norm = np.linalg.norm(v)
        by_doc_vecs[doc_i] = v / norm if norm > 0 else v
# вариант для всех доков кроме пдф
    # candidate_ids = sorted(
    #     doc_i for doc_i in by_doc_vecs.keys()
    #     if doc_i < len(docs) and not _is_pdf_source(docs[doc_i])
    # )
    # skipped_pdf = len(by_doc_vecs) - len(candidate_ids)

# вариант только для веб-страниц
    candidate_ids = sorted(
        doc_i for doc_i in by_doc_vecs.keys()
        if doc_i < len(docs) and _is_web_page_source(docs[doc_i])
    )
    skipped_non_web = len(by_doc_vecs) - len(candidate_ids)

    def _doc_label(doc_i):
        d = docs[doc_i]
        fp = str(d.get("file_path", "") or "").split("/")[-1]
        url = str(d.get("url", "") or "")
        return f"{fp}" + (f" | {url}" if url else "")

    dropped = set()
    merge_log = []

    for i_pos, di in enumerate(candidate_ids):
        if di in dropped:
            continue
        for dj in candidate_ids[i_pos + 1:]:
            if dj in dropped:
                continue
            sim = float(by_doc_vecs[di] @ by_doc_vecs[dj])

            if sim < threshold:
                continue
            if _dates_conflict(
                doc_dates.get(di), doc_dates.get(dj),
                file_path_a=docs[di].get("file_path", ""),
                file_path_b=docs[dj].get("file_path", ""),
            ):
                continue

            loser = di if doc_lens.get(di, 0) < doc_lens.get(dj, 0) else dj
            winner = dj if loser == di else di
            dropped.add(loser)

            merge_log.append(
                f"      sim={sim:.4f} | keep=[{winner}] {_doc_label(winner)} "
                f"| drop=[{loser}] {_doc_label(loser)}"
            )

    # if dropped or skipped_pdf:
    if dropped or skipped_non_web:
        print(f"    [dedupe_docs] похожих документов отброшено: {len(dropped)} "
              f"(threshold={threshold}, date_tolerance_days={RAG_DOC_DEDUPE_DATE_TOLERANCE_DAYS}) | "
            #   f"PDF исключено из рассмотрения: {skipped_pdf}")
            f"PDF исключено из рассмотрения: {skipped_non_web}")
        for line in merge_log:
            print(line)

    return [ch for ch in chunks if ch["doc_i"] not in dropped]

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
    prompt1 = build_stage_prompt(
        company_id, domain, inn, context1, 1,
        STAGE1_VARIABLES, STAGE1_SYSTEM_PROMPT
    )
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


if __name__ == "__main__":
    print("Конфигурация загружена")
    print("SOURCES_DIR:", SOURCES_DIR)
    print("OUTPUT_CSV:", OUTPUT_CSV)
    print("GATEWAY_URL:", GATEWAY_URL)
    run_rag_by_variable(
        output_csv=OUTPUT_CSV,
        full_context_limit=int(os.environ.get("RAG_FULL_CONTEXT_LIMIT", "80000")),
        recall_k=int(os.environ.get("RAG_RECALL_K", "48")),
        final_k=int(os.environ.get("RAG_FINAL_K", "20")),
        max_chars_per_chunk=int(os.environ.get("RAG_MAX_CHARS_PER_CHUNK", "1600")),
        sleep_s=float(os.environ.get("RAG_SLEEP_S", "0.0")),
    )