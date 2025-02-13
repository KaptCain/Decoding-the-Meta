import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

#  Map URL names to actual map names
map_name_mapping = {
    "Infinity": "Abyss",
    "Duality": "Bind",
    "Canyon": "Fracture",
    "Triad": "Haven",
    "Jam": "Lotus",
    "Pitt": "Pearl",
    "Bonsai": "Split"
}

#  List of Maps (URL names)
maps = list(map_name_mapping.keys())

#  List of Ranks
ranks = ["ALL", "Iron", "Bronze", "Silver", "Gold", "Platinum", "Diamond", "Ascendant", "Immortal", "Radiant"]

#  Configure Selenium (Headless Mode)
options = Options()
options.add_argument("--headless")  
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

#  Start WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

#  Data storage
data = []

#  Loop through all maps and ranks
for map_url_name in maps:
    for rank in ranks:
        url = f"https://www.vstats.gg/agents?table=agents&map={map_url_name}&rank={rank}"
        print(f"Scraping: {url}")

        # Open URL in Selenium
        driver.get(url)

        try:
            #  Wait for table to load (up to 10 seconds)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tr[data-v-27180d48]"))
            )
        except:
            print(f" No agent data found for {map_url_name} - {rank}. Skipping...")
            continue

        #  Find agent rows
        rows = driver.find_elements(By.CSS_SELECTOR, "tr[data-v-27180d48]")

        for row in rows:
            print(f" Row Text: {row.text}")  

            try:
        #  Extract the entire row text and split into parts
                row_data = row.text.split()
                print(f" Parsed Row Data: {row_data}")

                if len(row_data) < 4:
                    print(f" Not enough data in row: {row_data}. Skipping.")
                    continue

        #  Extract agent name (assuming the second element in the row)
                agent_name = row_data[1]

        #  Extract role from the div with class "v-responsive v-img"
                try:
                    role_element = row.find_element(By.CSS_SELECTOR, "div.v-responsive.v-img")
                    role = role_element.get_attribute("aria-label") if role_element else "Unknown"
                except:
                    role = "Unknown"

        #  Extract win rate (first percentage found)
                win_rate = None
                for item in row_data:
                    if "%" in item:  
                        win_rate = float(item.replace("%", ""))
                        break  

                if win_rate is None:
                    print(f" No win rate found in row: {row_data}. Skipping.")
                    continue

        #  Extract total matches (last number in row_data)
                total_matches = None
                for item in reversed(row_data):  
                    if item.replace(",", "").isdigit():  
                        total_matches = int(item.replace(",", ""))
                        break

                if total_matches is None:
                    print(f" No total matches found in row: {row_data}. Skipping.")
                    continue

        #  Convert map name from URL name to actual name
                actual_map_name = map_name_mapping.get(map_url_name, map_url_name)

        #  Store data
                data.append((actual_map_name, rank, agent_name, role, win_rate, total_matches))

        #  Print sample data
                print(f" Scraped: {actual_map_name}, {rank}, {agent_name}, {role}, {win_rate}%, {total_matches} matches")

            except Exception as e:
                print(f" Error scraping {map_url_name} - {rank}: {str(e)}")
                continue
        #  Delay to avoid getting blocked
        time.sleep(2)

#  Close browser
driver.quit()

#  If no data was scraped, print an error
if not data:
    print(" No data scraped. Check debug_page.html to see if elements are missing.")
else:
    #  Create Pandas DataFrame
    columns = ["Map", "Rank", "Agent", "Role", "WinRate", "TotalMatches"]
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV (overwrite mode)
    output_path = "C:/Users/chase/class/Decoding-the-Meta/valorant_agent_stats.csv"
    df.to_csv(output_path, index=False)

print(" Scraping Completed! Data saved at:", output_path)
