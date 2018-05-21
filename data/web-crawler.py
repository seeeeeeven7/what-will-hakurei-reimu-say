import requests
from bs4 import BeautifulSoup
import re

allfile = open('talk_in_game/all.txt', 'w', encoding = 'utf8')

def crawl(url, save):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html5lib")

    outfile = open('talk_in_game/' +save, 'w', encoding = 'utf8')

    items = soup.find_all('tr', class_='tt-content')
    for item in items:
        speaker = item.find('div', class_='poem').get_text()
        if speaker == '博丽灵梦':
            s = str(item.find('td', class_='tt-zh')).replace('\n', '')[46:-11]
            s = s.replace('<p>', '').replace('</p>', '')
            s = s.replace('<br/><br/>', '\n').replace('<br/>', ' ')
            s = re.sub(r'(?is)</html>.+', '</html>', s)
            print(s, file = outfile)
            print(s, file = allfile)

# crawl("https://thwiki.cc/游戏对话:东方红魔乡/博丽灵梦", "TH06.txt")
# crawl("https://thwiki.cc/游戏对话:东方红魔乡/博丽灵梦_ExStory", "TH06EX.txt")

# crawl("https://thwiki.cc/游戏对话:东方妖妖梦/博丽灵梦", "TH07.txt")
# crawl("https://thwiki.cc/游戏对话:东方妖妖梦/博丽灵梦_ExStory", "TH07EX.txt")
# crawl("https://thwiki.cc/游戏对话:东方妖妖梦/博丽灵梦_PhStory", "TH07PH.txt")

# crawl("https://thwiki.cc/游戏对话:东方永夜抄/幻想的结界组", "TH08.txt")
# crawl("https://thwiki.cc/游戏对话:东方永夜抄/幻想的结界组_ExStory", "TH08EX.txt")

# crawl("https://thwiki.cc/游戏对话:东方花映冢/博丽灵梦", "TH09.txt")
# crawl("https://thwiki.cc/游戏对话:东方花映冢/博丽灵梦/对战", "TH09BA.txt")

# crawl("https://thwiki.cc/游戏对话:东方风神录/博丽灵梦", "TH10.txt")
# crawl("https://thwiki.cc/游戏对话:东方风神录/博丽灵梦_ExStory", "TH10EX.txt")

# crawl("https://thwiki.cc/游戏对话:东方地灵殿/博丽灵梦（八云紫支援）", "TH11BYZ.txt")
# crawl("https://thwiki.cc/游戏对话:东方地灵殿/博丽灵梦（八云紫支援）_ExStory", "TH11BYZEX.txt")
# crawl("https://thwiki.cc/游戏对话:东方地灵殿/博丽灵梦（伊吹萃香支援）", "TH11YCCX.txt")
# crawl("https://thwiki.cc/游戏对话:东方地灵殿/博丽灵梦（伊吹萃香支援）_ExStory", "TH11YCCXEX.txt")
# crawl("https://thwiki.cc/游戏对话:东方地灵殿/博丽灵梦（射命丸文支援）", "TH11SMWW.txt")
# crawl("https://thwiki.cc/游戏对话:东方地灵殿/博丽灵梦（射命丸文支援）_ExStory", "TH11SMWWEX.txt")

# crawl("https://thwiki.cc/游戏对话:东方星莲船/博丽灵梦A", "TH12A.txt")
# crawl("https://thwiki.cc/游戏对话:东方星莲船/博丽灵梦A_ExStory", "TH12AEX.txt")
# crawl("https://thwiki.cc/游戏对话:东方星莲船/博丽灵梦B", "TH12B.txt")
# crawl("https://thwiki.cc/游戏对话:东方星莲船/博丽灵梦B_ExStory", "TH12BEX.txt")

# crawl("https://thwiki.cc/游戏对话:东方神灵庙/博丽灵梦", "TH13.txt")
# crawl("https://thwiki.cc/游戏对话:东方神灵庙/博丽灵梦_ExStory", "TH13EX.txt")

# crawl("https://thwiki.cc/游戏对话:东方辉针城/博丽灵梦A", "TH14A.txt")
# crawl("https://thwiki.cc/游戏对话:东方辉针城/博丽灵梦A_ExStory", "TH14AEX.txt")
# crawl("https://thwiki.cc/游戏对话:东方辉针城/博丽灵梦B", "TH14B.txt")
# crawl("https://thwiki.cc/游戏对话:东方辉针城/博丽灵梦B_ExStory", "TH14BEX.txt")

crawl("https://thwiki.cc/游戏对话:东方绀珠传/博丽灵梦", "TH15.txt")
crawl("https://thwiki.cc/游戏对话:东方绀珠传/博丽灵梦_ExStory", "TH15EX.txt")

crawl("https://thwiki.cc/游戏对话:东方天空璋/博丽灵梦", "TH16.txt")
crawl("https://thwiki.cc/游戏对话:东方天空璋/博丽灵梦_ExStory", "TH16EX.txt")