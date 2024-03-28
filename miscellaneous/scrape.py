import playwright.sync_api as pw
import re

with pw.sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(
        "https://prd-tnm.s3.amazonaws.com/index.html?prefix=StagedProducts/Elevation/13/TIFF/current/"
    )
    page.wait_for_selector("#listing")
    page.screenshot(path="test.png")
    links = page.locator("#listing a")
    with open("dem_paths.csv", "w") as output:
        for i in range(1, links.count()):
            link = str(links.nth(i).get_attribute("href"))
            name = link[-(link[:-1][::-1].index("/") + 1) : -1]
            print(name)
            output.write(f"{name}/USGS_13_{name}.tif,\n")

    browser.close()
