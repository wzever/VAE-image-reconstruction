import torch
from main import *
pretrained = 'pretrained/05-23_01_02-1000.pt'

if not os.path.exists('fusionResult'):
        os.makedirs('fusionResult')

old_man = ['dataset/lfw/Abdullah_al-Attiyah/Abdullah_al-Attiyah_0001.jpg',
            'dataset/lfw/Tom_Harkin/Tom_Harkin_0005.jpg',
            'dataset/lfw/Rowan_Williams/Rowan_Williams_0001.jpg',
            'dataset/lfw/Billy_Graham/Billy_Graham_0002.jpg',
            'dataset/lfw/Bob_Graham/Bob_Graham_0004.jpg',
            'dataset/lfw/Emilio_Botin/Emilio_Botin_0001.jpg']
young_man = ['dataset/lfw/Linus_Roache/Linus_Roache_0001.jpg', 
                'dataset/lfw/Liu_Ye/Liu_Ye_0001.jpg',
                'dataset/lfw/Ray_Young/Ray_Young_0001.jpg',
                'dataset/lfw/Zach_Parise/Zach_Parise_0001.jpg',
                'dataset/lfw/Stanley_Tong/Stanley_Tong_0001.jpg']
old_woman = ['dataset/lfw/Lily_Safra/Lily_Safra_0001.jpg',
                'dataset/lfw/Elizabeth_Taylor/Elizabeth_Taylor_0001.jpg',
                'dataset/lfw/Emma_Nicholson/Emma_Nicholson_0001.jpg',
                'dataset/lfw/Julie_Gerberding/Julie_Gerberding_0007.jpg',
                'dataset/lfw/Mary_Robinson/Mary_Robinson_0001.jpg']
young_woman = ['dataset/lfw/Lesley_Flood/Lesley_Flood_0001.jpg', 
                'dataset/lfw/Lisa_Ling/Lisa_Ling_0001.jpg',
                'dataset/lfw/Jessica_Lynch/Jessica_Lynch_0002.jpg',
                'dataset/lfw/Giulietta_Masina/Giulietta_Masina_0001.jpg',
                'dataset/lfw/Elizabeth_Smart/Elizabeth_Smart_0003.jpg']

for paths in [young_man, old_man, old_woman, young_woman]:
    random.shuffle(paths)
X = young_man + young_woman
Y = old_man[:5] + old_woman
X = torch.from_numpy(load_image(X))
Y = torch.from_numpy(load_image(Y))

im_show = Y

for i in range(11):
    alpha = i / 10
    new_img = gen_feature_fusion(X, Y, alpha, pretrained)
    im_show = torch.cat([im_show, new_img], dim=3)
im_show = torch.cat([im_show, X], dim=3)
save_image(im_show, 'fusionResult/old_young.png', nrow=1)
