from playwright.sync_api import Playwright, sync_playwright, expect
from time import sleep
import random

def run(playwright: Playwright, img_url) -> None:
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto("https://www.google.com/imghp?sbi=1")
    sleep(2)
    # 检查“全部拒绝”按钮是否存在
    reject_buttons = page.query_selector_all('text="全部拒绝"')

    if reject_buttons:
        # 如果存在“全部拒终”按钮，点击第一个找到的按钮
        reject_buttons[0].click()
        print("点击了‘全部拒绝’按钮。")
        sleep(1)
        page.get_by_label("按图搜索").click()
    else:
        # 如果没有找到按钮，不执行任何操作
        print("没有找到‘全部拒绝’按钮。")

    page.get_by_label("按图搜索").click()

    input_selector = 'css=input[type="file"]'
    page.set_input_files(input_selector, img_url)
    tmp_cnt = random.randint(5,10)
    sleep(tmp_cnt)

    # 获取页面中所有class属性为“G19kAf ENn9pd”的div标签aa
    div_aa_elements = page.query_selector_all('.G19kAf.ENn9pd')

    # 准备用于保存结果的列表
    results = []

    for div_aa in div_aa_elements:
        # 获取每一个div标签aa内class为“Vd9M6”的div标签bb
        div_bb = div_aa.query_selector('.Vd9M6')
        if div_bb:
            # 获取div标签bb内<a>标签的href和aria-label的值
            a_element = div_bb.query_selector('a')
            if a_element:
                link = a_element.get_attribute('href')
                label = a_element.get_attribute('aria-label')
                # 将结果保存到列表中
                results.append({'thumbnail': link, 'title': label})

    bt = page.get_by_role("button", name="查找图片来源")
    if bt:
        bt.click()
    else:
        bt = page.get_by_role("button", name="Find image source")
        if bt:
            bt.click()
    tmp_cnt = random.randint(10,15)
    sleep(tmp_cnt)
    try:
        li_elements = page.query_selector_all('div[jsslot] ul > li')
    except TimeoutError as e:
        print(e)
        page.close()
        context.close()
        browser.close()
        # 打印结果
        print(results)
        data = {'num': len(results), 'data': results}
        return data


    link_list = []
    label_list = []
    for li in li_elements:
        # 获取 <li> 内的 <a> 标签
        a_element = li.query_selector('a')
        if a_element:
            # 获取 <a> 标签的 href 和 aria-label 属性
            href = a_element.get_attribute('href')
            link_list.append(href)
            aria_label = a_element.get_attribute('aria-label')
            label_list.append(aria_label)
            print(f"href: {href}, aria-label: {aria_label}")
        else:
            print("No <a> tag found in this <li>.")

    num_res = len(link_list)
    data = {'num': num_res, 'data': []}
    for i in range(num_res):
        data['data'].append({'thumbnail': link_list[i], 'title': label_list[i]})
    if data['num'] == 0:
        data['num'] = len(results)
        data['data'] = results

    page.close()

    # ---------------------
    context.close()
    browser.close()
    return data

if __name__ == "__main__":
    img_url = 'data\\pic\\0a8799dd7f14ef91.jpg'
    data = {}
    with sync_playwright() as playwright:
        data = run(playwright, img_url)
    print(data)