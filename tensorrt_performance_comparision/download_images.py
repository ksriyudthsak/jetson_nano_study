import os
import urllib.request

if not os.path.isdir("data"):
    os.makedirs("data") 

# download image samples
sample_list = [
"https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/440px-Siam_lilacpoint.jpg",
"https://media.wired.com/photos/5bb532b7f8a2e62d0bd5c4e3/master/w_2560%2Cc_limit/bee-146810332.jpg",
"https://image-cdn.hypb.st/https%3A%2F%2Fhypebeast.com%2Fimage%2F2020%2F05%2Fnba-drops-spalding-basketball-for-wilson-2021-season-1.jpg?q=90&w=1400&cbr=1&fit=max",
"http://www.kamogawa-seaworld.jp/aquarium/aquarium_info/images/kurage201908/ph01.jpg",
"https://files.nccih.nih.gov/horse-chestnut-aesculus-hippocastanum-gettyimages-95791857-square.jpg"
,
"https://zukan.pokemon.co.jp/zukan-api/up/images/index/0307cb5bf57607937c2112f9791eb7dd.png",
"https://zukan.pokemon.co.jp/zukan-api/up/images/index/481d67bad0e4899717bb69f08b4c1a94.png",
"https://img.gamewith.jp/article/thumbnail/rectangle/25235.png?date=1555917396",
"https://zukan.pokemon.co.jp/zukan-api/up/images/index/39e9311c11f99b98d9a8c3f8389470fc.png",
"https://zukan.pokemon.co.jp/zukan-api/up/images/index/e4ff08120b4329ddef9896c4b44a6cf0.png"
]

for i in range(len(sample_list)):
    with urllib.request.urlopen(sample_list[i]) as u:
        with open('./data/img{}.jpg'.format(i), 'wb') as o:
            o.write(u.read())

# download imagenet labels
imagenet_classes = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
with urllib.request.urlopen(imagenet_classes) as u:
    with open('./data/imagenet_classes.txt', 'wb') as o:
        o.write(u.read())
