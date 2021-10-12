from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from time import sleep
import os
import requests


chrome_path = "/Users/harunakanta/Desktop/aidemy_app/chromedriver"
options = Options()
options.add_argument("--incognito")
driver = webdriver.Chrome(executable_path=chrome_path, options=options)


member_name = ["テテ", "キム・テヒョン","ジン", "キム・ソクジン", "シュガ", 
               "ミン・ユンギ", "ジェイホープ", "チョン・ホソク", "アールエム", 
               "キム・ナムジュン","ジミン", "パク・ジミン", "ジョングク", 
               "チョン・ジョングク"]


for member in member_name:
    imag_urls = []
    url = "https://search.yahoo.co.jp/image"
    driver.get(url)
    search_box = driver.find_element_by_class_name("SearchBox__searchInput")
    search_box.send_keys(member + " BTS")
    search_box.submit()
    sleep(1)

    click_count = 0
    while click_count < 10:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(3)
        more_button = driver.find_element_by_class_name("sw-MoreButton")
        more_button.click()
        sleep(3)
        click_count += 1
        


    elements_1 = driver.find_elements_by_class_name("sw-Thumbnail")
    for element in elements_1:
        img = element.find_element_by_tag_name("img").get_attribute("src")
        imag_urls.append(img)
        sleep(1)


    IMAGE_DIR = "./yahho_images/"

    if os.path.isdir(IMAGE_DIR):
        print("already")
    else:
        os.makedirs(IMAGE_DIR)


    for i, imag_url in enumerate(imag_urls, start=1):
        image = requests.get(imag_url)
        file_name = member + "_" + str(i)
        with open(IMAGE_DIR + file_name + ".jpg", "wb") as f:
            f.write(image.content)
    sleep(1)
    
driver.quit()




