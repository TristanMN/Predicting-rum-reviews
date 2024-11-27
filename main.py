import requests
import pandas as pd
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import re

# #set directory to the ChromeDriver
# #chrome_options = Options()
# #chrome_options.add_argument("--headless")  # without graphic interface
# #chrome_options.add_argument('--disable-blink-features=AutomationControlled')
# #chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36")
# #driver = webdriver.Chrome(options=chrome_options)
# service = Service('C:\\ChromeDriver\\chromedriver-win64\\chromedriver.exe')
# driver = webdriver.Chrome(service=service)#, options=options)
#
# # Set the URL for the rum reviews page
# url = "https://rumratings.com/stream"
# driver.get(url)
#
# # # Loop to click "Show More" until no more pages are left
# # while True:
# #     try:
# #         # # Find the "Show More" button and click it
# #         # wait = WebDriverWait(driver, 10)
# #         # show_more_button = wait.until(
# #         #     EC.element_to_be_clickable((By.CLASS_NAME, "c-button--secondary"))
# #         # )
# #         # show_more_button.click()
# #         #
# #         # # Wait for the content to load (you may need to adjust the time based on the page load speed)
# #         # time.sleep(random.uniform(2, 5))
# #         print(iter)
# #         wait = WebDriverWait(driver, 10)
# #
# #         # waiting for element to be visible
# #         show_more_button = wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "c-button--secondary")))
# #
# #         # scroll to the element
# #         driver.execute_script("arguments[0].scrollIntoView();", show_more_button)
# #         time.sleep(1)
# #
# #         # click using JavaScript
# #         driver.execute_script("arguments[0].click();", show_more_button)
# #         time.sleep(random.uniform(2, 5))
# #         iter+=1
# #     except Exception as e:
# #         print(f"Error: {e}")
# #         # Break if the button is not found, indicating that there are no more pages
# #         break

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"
}

#translator = Translator()
translator = GoogleTranslator(source='auto', target='en')
df = pd.DataFrame(columns=['RumName', 'Rating', 'OpinionTitle', 'Opinion'])

for page in range(1, 401):
    url = f"https://rumratings.com/stream?fbclid=IwZXh0bgNhZW0CMTEAAR17zZK1w5IpesswHRzuKq7IgMKVx-Ilwi2q6CSer71kOLCw9sMyID7Z3IU_aem_m2iHrEeu6-4q-SskrI2etw&page={page}"
    # url = f"https://rumratings.com/stream?page={page}"
    response = requests.get(url, headers=headers)
    # print(response.raw)

    #page_source = driver.page_source
    if response.status_code == 200:
        #soup = BeautifulSoup(page_source, 'html.parser')
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the container with reviews
        reviews = soup.find_all('div', class_='stream-item')

        # Extract the opinions
        for i, review in enumerate(reviews):
            try:
                opinion_div = review.find('div', class_='review-container')
                opinion = opinion_div.find('p').get_text().replace('\n', ' ')
                #detected_language = translator.detect(opinion).lang
                #time.sleep(0.5)
                translated_opinion = translator.translate(opinion)

                name_div = review.find('div', class_='c-review-comment__posted-on')
                rum_name = name_div.find('a').get_text()
                rum_link = name_div.find('a').get('href')

                rating_div = review.find('div', class_='c-review-comment__content-header').find('h3')
                opinion_title = rating_div.find('div').get_text().strip()
                rating = rating_div.find('div', class_='rating-and-divider').find('span').get_text().strip()    #to delete white signs
                ratings_link = rating_div.find('a', text=re.compile(r'\d+ ratings'))
                if ratings_link:
                    user_review_count = int(ratings_link.text.split()[0])
                else:
                    user_review_count = 0

                # if detected_language != 'en':
                #     opinion = translator.translate(opinion, dest='en').text  #translation to the English language
                #     time.sleep(0.3)
                #     opinion_title = translator.translate(opinion_title, dest='en').text
                #     time.sleep(0.3)

                if translated_opinion != opinion:
                    opinion = translated_opinion
                    opinion_title = translator.translate(opinion_title)

                new_row = pd.DataFrame(
                    {
                    'RumName': rum_name,
                    'RumLink': rum_link,
                    'Rating': rating,
                    'OpinionTitle': opinion_title,
                    'Opinion': opinion,
                    'UserReviewCount': user_review_count,
                    # 'User': user_name,
                    },
                    index=[0])
                df = pd.concat([df, new_row], ignore_index=True)
            except AttributeError as e:
                print(f"Attribute error, page number {page}, opinion {i+1}: {e}")
                continue
    else:
        print(f"Failed to retrieve page with status code {response.status_code}")

    print(f"Progress done: page {page}/400")

#driver.quit()
df.to_csv('scrapped_data.csv', index=False)