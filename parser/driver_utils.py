import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def get_driver():
    chrome_options = Options()

    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    chrome_options.page_load_strategy = os.environ.get(
        "CHROME_PAGE_LOAD_STRATEGY", "eager"
    )

    chrome_binary = os.environ.get("CHROME_BINARY")
    if chrome_binary:
        chrome_options.binary_location = chrome_binary

    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation"]
    )
    chrome_options.add_experimental_option(
        "useAutomationExtension", False
    )

    driver = webdriver.Chrome(options=chrome_options)
    driver.set_page_load_timeout(
        int(os.environ.get("CHROME_PAGE_TIMEOUT", "60"))
    )
    return driver
