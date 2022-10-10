# -*- coding: utf-8 -*-

import sys
import time
import chromedriver_binary
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import re

"""
import 
pip install openpyxl
pip install selenium
pip install chromedriver-binary==104.0.5112.79
"""

# ログインページのURL
base_url = "https://www.min-inuzukan.com/"

# Webドライバ
driver = webdriver.Chrome()
driver.get(base_url) 
pattern = r'.*?/?([^/]+\.(png|jpg))'


def main():
    links = get_dog_links()
    for link in links:
        get_each_page_images(link)


def get_each_page_images(link):
    driver.get(link)
    img_links = set()
    
    element = driver.find_element(By.CLASS_NAME, 'mainImg')
    for img in element.find_elements(By.TAG_NAME, 'img'):
        img_links.add(img.get_attribute('src'))
    
    for img_link in img_links:
        response = requests.get(img_link)
        image = response.content
        _filename = img_link.replace(base_url + 'images/', '')
        filename = re.match(pattern, _filename)

        with open('fig/' + filename.group(1), "wb") as f:
            f.write(image)
        time.sleep(1)


def get_dog_links():
    lists = driver.find_elements(By.CLASS_NAME, 'list_6')
    links = set()
    for elements in lists:
        for element in elements.find_elements(By.TAG_NAME,'a'):
            links.add(element.get_attribute('href'))
    return links

"""
review_num = driver.find_element(By.XPATH, "//a[@href='{}']".format(review_url.replace(base_url, '')))
content = review_num.get_attribute('innerHTML').replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
time_tag = review.find_element(By.TAG_NAME, 'time')
datetime = time_tag.get_attribute("datetime")
profile = review.find_element(By.CSS_SELECTOR, ".mr-5.v-m").text
score = review.find_element(By.CSS_SELECTOR, ".ml-5.fs-14").text
content = review.find_element(By.CLASS_NAME, 'article_answer').get_attribute('innerHTML').replace('\n', '').replace('\r', '').replace('\t', '').replace(' ', '')
"""


                # 日付取得
                

if __name__ == '__main__':
    main()