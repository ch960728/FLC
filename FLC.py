import tkinter
from tkinter import *

import tkinter.ttk as ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
import numpy as np
import tkinter.messagebox as msgbox
import os


#분석용 함수들
#색 비율
def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
#내림차순으로 정렬
def hist_sort(hist):
    s=hist.argsort()#오름차순으로 정렬!
    hist = hist[s][::-1]#내림차순
    return hist

#bar형식으로 만들어주기
def plot_colors(hist, centroids):
    bar = np.zeros((50, 550, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 550)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar
def find_nearest(array, value):#가장 가까운 값 찾는 코드
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def image_Reshape(file):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    return image
#bar형식을 하나로 합치기 퍼센트 적용X
def plot_colors2(centroids):

    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for color in centroids:

        endX = startX + 30
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar
#np.zeros의 공백을 지워주자
def rebar(barsum):
    for k in range(0,50):
        barsum=np.delete(barsum,0,axis=0)
    return barsum

##===============================================================================================
root_ori = Tk()
root_ori.title("FLC")

root_ori.geometry("640x1000+600+5")  # 크기 + 위치
root_ori.resizable(False, False)  # 창 크기 변경 불가

root=Frame(root_ori)
root.pack()
btn_frame=Frame(root_ori)
btn_frame.pack(fill='both')
#1. 이미지 불러와서 게시하기
logo = Image.open(os.path.join(os.path.abspath('img'), 'logo2.png'))#로고 불러오기
logo = logo.resize((304,304)) #크기 조정
logo = ImageTk.PhotoImage(logo)
    #로고, 사용자 입력 이미지 출력 창
userimg = Label(btn_frame, image=logo)
userimg.pack(side='left')



#2. 이미지 이름 게시하기
userimgname = Label(btn_frame,text="검사할 이미지 이름",font=2,width=40, height=4) #사용자 입력 이미지 이름
userimgname.pack()

#3. 사용자 이미지 분석 게시하기
userimg_frame=Frame(root_ori)
userimg_frame.pack()
lv1_name = Label(userimg_frame,text="추출한 색상",fg='white', bg='#242424') #사용자 입력 이미지 이름
lv1_name.pack(side='left')

userimg_frame_img=Frame(root_ori)
userimg_frame_img.pack()
userimg_bar = Label(userimg_frame_img,bg="#242424", width=580, height=5)
userimg_bar.pack()

#4.분석용 이미지 게시하기
global checkliver1
checkliver1="img/1_barsumavg.png"
global checkliver2
checkliver2="img/2_barsumavg.png"
global checkliver3
checkliver3="img/3_barsumavg.png"
global checkliver4
checkliver4="img/4_barsumavg.png"
global lv1_file
global filenm
filenm = 'None'
global cutpage
cutpage = None
global currentImage
currentImage = {}


    #1단계 이미지, 이름
lv1_frame=Frame(root_ori)
lv1_frame.pack(fill='both')
lv1_img_frame=Frame(root_ori)
lv1_img_frame.pack()
lv1_name = Label(lv1_frame,text="1단계 분석 이미지",fg='white', bg='#242424') #사용자 입력 이미지 이름
lv1_name.pack(side='left',fill='both',padx=(30,0))
lv1_img = Image.open(os.path.join(os.path.abspath('img') , '1_barsumavg.png'))
lv1_img = lv1_img.resize((580,70)) #크기 조정
lv1_img = ImageTk.PhotoImage(lv1_img)
lv1 = Label(lv1_img_frame, image=lv1_img,bg='#242424') #출력



    #2단계 이미지, 이름
lv2_frame=Frame(root_ori)
lv2_frame.pack(fill='both')
lv2_img_frame=Frame(root_ori)
lv2_img_frame.pack()
lv2_name = Label(lv2_frame, text="2단계 분석 이미지", fg='white', bg='#242424') #사용자 입력 이미지 이름
lv2_name.pack(side='left',fill='both',padx=(30,0))
lv2_img = Image.open(os.path.join(os.path.abspath('img') , '2_barsumavg.png'))
lv2_img = lv2_img.resize((580,70)) #크기 조정
lv2_img = ImageTk.PhotoImage(lv2_img)
lv2 = Label(lv2_img_frame, image=lv2_img, bg='#242424') #출력



    #3단계 이미지, 이름
lv3_frame=Frame(root_ori)
lv3_frame.pack(fill='both')
lv3_img_frame=Frame(root_ori)
lv3_img_frame.pack()
lv3_name = Label(lv3_frame,text="3단계 분석 이미지", fg='white', bg='#242424') #사용자 입력 이미지 이름
lv3_name.pack(side='left',fill='both',padx=(30,0))
lv3_img = Image.open(os.path.join(os.path.abspath('img') , '3_barsumavg.png'))
lv3_img = lv3_img.resize((580,70)) #크기 조정
lv3_img = ImageTk.PhotoImage(lv3_img)
lv3 = Label(lv3_img_frame, image=lv3_img, bg='#242424') #출력



    #4단계 이미지, 이름
lv4_frame=Frame(root_ori)
lv4_frame.pack(fill='both')
lv4_img_frame=Frame(root_ori)
lv4_img_frame.pack()
lv4_name = Label(lv4_frame,text="4단계 분석 이미지",fg='white',bg='#242424') #사용자 입력 이미지 이름
lv4_name.pack(side='left',fill='both',padx=(30,0))
lv4_img = Image.open(os.path.join(os.path.abspath('img') , '4_barsumavg.png'))#로고 게시
lv4_img = lv4_img.resize((580,70)) #크기 조정
lv4_img = ImageTk.PhotoImage(lv4_img)
lv4 = Label(lv4_img_frame, image=lv4_img,bg='#242424') #출력




#이미지 자르기 전 파일 불러 왔는지 확인
def file_ok():
    if filenm=='None':
        print('파일 없음')
        msgbox.showwarning("알림", "이미지를 불러와 주세요")
    else:
        imgcut_page()

#이미지 자르기 위한 새 창-------------------------------------------------
def imgcut_page():
    topx, topy, botx, boty = 0, 0, 0, 0
    global rect_id
    rect_id = None

    def get_mouse_posn(event):
        global topy, topx

        topx, topy = event.x, event.y

    def update_sel_rect(event):
        global rect_id
        global topy, topx, botx, boty

        botx, boty = event.x, event.y
        #print(topx, topy, botx, boty)
        global box
        global w,h
        if w+h>2000:
            box = (topx*2, topy*2, botx*2, boty*2)
        else:
            box = (topx, topy, botx, boty)
        canvas.coords(rect_id, topx, topy, botx, boty)# Update selection rect.
    global cutpage
    cutpage = tkinter.Toplevel()
    cutpage.title("이미지 자르기")
    global roi
    changed_file5 = Image.open(user_file)
    roi = changed_file5
    global photo_cutimg
    photo_cutimg = ImageTk.PhotoImage(changed_file5)
    global w,h,w2,h2
    w = int(photo_cutimg.width())
    h = int(photo_cutimg.height())
    w2 = int(w/2)
    h2 = int(h/2)

    #캔버스
    if w + h > 2000:
        changed_file5 = Image.open(user_file)
        changed_file5 = changed_file5.resize((w2, h2))
        photo_cutimg = ImageTk.PhotoImage(changed_file5)
        cutpage.geometry('%sx%s' % (w2, (h2 + 100)))  # 크기 + 위치
        cutpage.resizable(True, True)  # 창 크기 변경 불가
        cut_img_name = Label(cutpage, text=filenm[1], fg='white', bg='#242424',width=w2,height=2)  # 사용자 입력 이미지 이름
        cut_img_name.pack()
        canvas = tkinter.Canvas(cutpage, width=w2, height=h2,
                                borderwidth=0, highlightthickness=0)  # 이미지를 자르기 위해 캔버스로 불러와 줌
        canvas.pack()
        canvas.create_image(0,0, anchor="nw",image=photo_cutimg)
        rect_id = canvas.create_rectangle(topx, topy, topx, topy,
                                          dash=(2, 2), fill='', outline='red')
    else:
        cutpage.geometry('%sx%s' % (w, (h + 100)))  # 크기 + 위치
        cutpage.resizable(True, True)  # 창 크기 변경 불가
        cut_img_name = Label(cutpage, text=filenm[1], fg='white',font=2, bg='#242424',width=w,height=2)  # 사용자 입력 이미지 이름
        cut_img_name.pack()
        canvas = tkinter.Canvas(cutpage, width=photo_cutimg.width(), height=photo_cutimg.height(),
                       borderwidth=0, highlightthickness=0)#이미지를 자르기 위해 캔버스로 불러와 줌
        canvas.pack()
        canvas.create_image(0,0, anchor="nw",image=photo_cutimg)
        rect_id = canvas.create_rectangle(topx, topy, topx, topy,
                                          dash=(2, 2), fill='', outline='red')


    def imagecut():#이미지 자르기
        global roi
        global w,h,w2,h2
        canvas.delete("all")

        if w+h>2000:#이미지가 큰 경우
            roi = currentImage['data'].crop(box)
            roi = roi.resize((w2, h2))#잘라낸 이미지를 크기에 맞춤
            currentImage['data'] = roi
            changed_cut = ImageTk.PhotoImage(roi)
            canvas.create_image(0,0,anchor="nw", image=changed_cut)
            canvas.image = changed_cut
            w = int(changed_cut.width())
            h = int(changed_cut.height())


        else:
            roi = currentImage['data'].crop(box)
            roi = roi.resize((w, h))
            currentImage['data'] = roi
            changed_cut = ImageTk.PhotoImage(roi)
            canvas.create_image(0,0,anchor="nw", image=changed_cut)
            canvas.image = changed_cut
            w = int(changed_cut.width())
            h = int(changed_cut.height())


        global rect_id
        rect_id = None
        rect_id = canvas.create_rectangle(topx, topy, topx, topy,
                                           dash=(2, 2), fill='', outline='red')


        return currentImage


    def save_pic():#이미지 저장하기
        result = filedialog.asksaveasfilename(initialdir="/", title="Select file",defaultextension='.png',filetypes=([('PNG 파일', '*.png')]))
        if result:
            print(result)
            roi.save(result)



    imgcut_btn = Button(cutpage, text="이미지 자르기", bg='#575757', font=2, fg='white', width=13, height =10, command=imagecut)
    imgcut_btn.pack(side='left',expand=True,fill="both")
    imgsave_btn = Button(cutpage, text="이미지 저장하기", bg='#575757',font=2, fg='white', width=13, height =10,command = save_pic)
    imgsave_btn.pack(side='right',expand=True, fill="both")

    canvas.bind('<Button-1>', get_mouse_posn)
    canvas.bind('<B1-Motion>', update_sel_rect)

#----------------------------------
def userimgload(): # 사용자 이미지 입력시, 사진과 이름 출력 해줌
    global cutpage
    if cutpage:
        cutpage.destroy()
    global user_file #사용자가 불러온 이미지를 다른 함수에서도 쓰기 위해 global로 지정함
    user_file = filedialog.askopenfilename(title="이미지 파일을 선택하세요",filetypes=(("PNG 파일","*.png"),("jpg","*.jpg")),initialdir="C:/")#파일 이름 가져옴
    if user_file:
        global filenm
        filenm = os.path.split(user_file)
        userimgname.config(text=filenm[1],font=2)
        global photo_userimg2#전역변수로 지정해주어야 바뀌어도 값이 사라지지 않음
        changed_file = Image.open(user_file)
        currentImage['data'] = changed_file
        changed_file = changed_file.resize((304,304))
        photo_userimg2 = ImageTk.PhotoImage(changed_file)
        userimg.config(image=photo_userimg2)
        print(currentImage['data'])
        listbox.delete(0,END)
    else:
        userimgname.config(text="검사할 이미지 이름")
    return user_file

#분석 이미지 불러오기
def lv1load(): # 사용자 이미지 입력시, 사진과 이름 출력 해줌
    global lv1_file #사용자가 불러온 이미지를 다른 함수에서도 쓰기 위해 global로 지정함
    global filenm1
    lv1_file = filedialog.askopenfilename(title="이미지 파일을 선택하세요",filetypes=(("PNG 파일","*.png"),("jpg","*.jpg")),initialdir="C:/")#파일 이름 가져옴
    filenm1 = os.path.split(lv1_file)
    lv1_name.config(text='1단계 : '+filenm1[1])
    global photo_lv1img2#전역변수로 지정해주어야 바뀌어도 값이 사라지지 않음
    changed_file = Image.open(lv1_file)
    changed_file = changed_file.resize((580,70))
    photo_lv1img2 = ImageTk.PhotoImage(changed_file)
    lv1.config(image=photo_lv1img2)
    listbox.insert(0, '1단계 선택한 이미지'+filenm1[1])
    global checkliver1
    checkliver1 = lv1_file
    return lv1_file
def lv2load(): # 사용자 이미지 입력시, 사진과 이름 출력 해줌
    global lv2_file #사용자가 불러온 이미지를 다른 함수에서도 쓰기 위해 global로 지정함
    global filenm2
    lv2_file = filedialog.askopenfilename(title="이미지 파일을 선택하세요",filetypes=(("PNG 파일","*.png"),("jpg","*.jpg")),initialdir="C:/")#파일 이름 가져옴
    filenm2 = os.path.split(lv2_file)
    lv2_name.config(text='2단계 : '+filenm2[1])
    global photo_lv2img2#전역변수로 지정해주어야 바뀌어도 값이 사라지지 않음
    changed_file = Image.open(lv2_file)
    changed_file = changed_file.resize((580,70))
    photo_lv2img2 = ImageTk.PhotoImage(changed_file)
    lv2.config(image=photo_lv2img2)
    listbox.insert(0, '2단계 선택한 이미지'+filenm2[1])
    global checkliver2
    checkliver2 = lv2_file
    return lv2_file
def lv3load(): # 사용자 이미지 입력시, 사진과 이름 출력 해줌
    global lv3_file #사용자가 불러온 이미지를 다른 함수에서도 쓰기 위해 global로 지정함
    global filenm3
    lv3_file = filedialog.askopenfilename(title="이미지 파일을 선택하세요",filetypes=(("PNG 파일","*.png"),("jpg","*.jpg")),initialdir="C:/")#파일 이름 가져옴
    filenm3 = os.path.split(lv3_file)
    lv3_name.config(text='3단계 : '+filenm3[1])
    global photo_lv1img3#전역변수로 지정해주어야 바뀌어도 값이 사라지지 않음
    changed_file = Image.open(lv3_file)
    changed_file = changed_file.resize((580,70))
    photo_lv1img3 = ImageTk.PhotoImage(changed_file)
    lv3.config(image=photo_lv1img3)
    listbox.insert(0, '3단계 선택한 이미지'+filenm3[1])
    global checkliver3
    checkliver3 = lv3_file
    return lv3_file
def lv4load(): # 사용자 이미지 입력시, 사진과 이름 출력 해줌
    global lv4_file #사용자가 불러온 이미지를 다른 함수에서도 쓰기 위해 global로 지정함
    global filenm4
    lv4_file = filedialog.askopenfilename(title="이미지 파일을 선택하세요",filetypes=(("PNG 파일","*.png"),("jpg","*.jpg")),initialdir="C:/")#파일 이름 가져옴
    filenm4 = os.path.split(lv4_file)
    lv4_name.config(text='4단계 : '+filenm4[1])
    global photo_lv1img4#전역변수로 지정해주어야 바뀌어도 값이 사라지지 않음
    changed_file = Image.open(lv4_file)
    changed_file = changed_file.resize((580,70))
    photo_lv1img4 = ImageTk.PhotoImage(changed_file)
    lv4.config(image=photo_lv1img4)
    listbox.insert(0, '4단계 선택한 이미지'+filenm4[1])
    global checkliver4
    checkliver4 = lv4_file
    return lv4_file

def checkliver():# 사용자 이미지 검사
    print(user_file)#사용자 파일 경로 가져옴
    p_var2.set(5)
    progress_bar2.update()
    image = cv2.imread(user_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    k = 30  # 색상 K개 추출
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    hist = centroid_histogram(clt)
    colorSort = clt.cluster_centers_
    bar = plot_colors(hist, colorSort)
    global photo_userimgcheck  # 사용자 check이미지를 사용하기 위해서 global로 해주어야 함
    usercheckimg = Image.fromarray(bar)#넘파이를 이미지로 바꿔줌
    photo_userimgcheck = ImageTk.PhotoImage(image=usercheckimg)  # bar을 pilimage로 바꿔주고 tk로 열어주어야함
    userimg_bar.config(image=photo_userimgcheck,width=580,height=70)
    # #이미지 분석하는 코드
    img_original = bar#원본
    img1 = cv2.imread(checkliver1)#1단계
    img2 = cv2.imread(checkliver2)#2단계
    img3 = cv2.imread(checkliver3)#3단계
    img4 = cv2.imread(checkliver4)#4단계
    p_var2.set(90)
    progress_bar2.update()

    #원본 정규화
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
    img_original = cv2.calcHist([img_original], [0, 1], None, [180, 256], [0, 180, 0, 256])
    img_original = cv2.normalize(img_original, None, 0, 1, cv2.NORM_MINMAX)

    imgs = [img1, img2, img3, img4]# 단계 사진만 구분해서 정규화함
    hists = []
    for i, img in enumerate(imgs):
        #1 각 이미지를 HSV로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #2 H,S 채널에 대한 히스토그램 계산
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        #3 0~1로 정규화
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hists.append(hist)

    #query = hists[0]#원본 이미지

    correl_result = []
    chisqr_result = []
    intersect_result = []
    bhattacharyya_result = []

    correl = cv2.HISTCMP_CORREL
    chisqr = cv2.HISTCMP_CHISQR
    intersect = cv2.HISTCMP_INTERSECT # ret = ret / np.sum(query) , 교차 분석인 경우 비교대상으로 나누어 1로 정규화
    bhattacharyya = cv2.HISTCMP_BHATTACHARYYA
    listbox.insert(0, '선택한 이미지' + filenm[1])
    #코렐 분석
    for hist in hists:
        c_rs = cv2.compareHist(img_original, hist, correl)
        correl_result.append(c_rs)

    correl_result = np.array(correl_result)
    correl_similar = find_nearest(correl_result, 1.0)#1.0과 가장 가까운 값 찾음
    correl_result = correl_result.tolist()
    correl_similar_index = correl_result.index(correl_similar)#가장 비슷한 이미지 위치 반환
    #print('해당 이미지는',correl_similar_index+1,'단계 입니다.')
    listbox.insert(1, '코렐분석 결과 해당 이미지는' + str(correl_similar_index+1)+' 단계 입니다.')

    #카이제곱
    for hist in hists:
        ch_rs = cv2.compareHist(img_original, hist, chisqr)
        chisqr_result.append(ch_rs)

    chisqr_result = np.array(chisqr_result)
    chisqr_similar = find_nearest(chisqr_result, 0)#0과 가장 가까운 값 찾음
    chisqr_result = chisqr_result.tolist()
    chisqr_similar_index = chisqr_result.index(chisqr_similar)#가장 비슷한 이미지 위치 반환
    #print('해당 이미지는',correl_similar_index+1,'단계 입니다.')
    listbox.insert(2, '카이제곱분석 결과 해당 이미지는' + str(chisqr_similar_index+1)+' 단계 입니다.')

    #교차검증
    for hist in hists:
        intersect_rs = cv2.compareHist(img_original, hist, intersect)
        intersect_result.append(intersect_rs/np.sum(img_original))

    intersect_result = np.array(intersect_result)
    intersect_similar = find_nearest(intersect_result, 1.0)#1.0과 가장 가까운 값 찾음
    intersect_result = intersect_result.tolist()
    intersect_similar_index = intersect_result.index(intersect_similar)#가장 비슷한 이미지 위치 반환
    #print('해당 이미지는',correl_similar_index+1,'단계 입니다.')
    listbox.insert(3, '교차검증 결과 해당 이미지는' + str(intersect_similar_index+1)+' 단계 입니다.')

    #바타차야 거리
    for hist in hists:
        bhattacharyya_rs = cv2.compareHist(img_original, hist, bhattacharyya)
        bhattacharyya_result.append(bhattacharyya_rs)

    bhattacharyya_result = np.array(bhattacharyya_result)
    bhattacharyya_similar = find_nearest(bhattacharyya_result, 0)#0과 가장 가까운 값 찾음
    bhattacharyya_result = bhattacharyya_result.tolist()
    bhattacharyya_similar_index = bhattacharyya_result.index(bhattacharyya_similar)#가장 비슷한 이미지 위치 반환
    #print('해당 이미지는',correl_similar_index+1,'단계 입니다.')
    listbox.insert(4, '바타차야 거리 분석 결과 해당 이미지는' + str(bhattacharyya_similar_index+1)+' 단계 입니다.')

    #총 갯수 출력해주기
    total_list = [correl_similar_index,chisqr_similar_index,intersect_similar_index,bhattacharyya_similar_index]
    total_list = np.array(total_list)
    total_list_count = np.bincount(total_list)#토탈 값 카운팅
    print(total_list_count)
    total_list_max = np.max(total_list_count)# 카운팅 중 가장 높은 수
    print(total_list_max)
    total_list_count = total_list_count.tolist()
    total_list_index = total_list_count.index(total_list_max)
    print(total_list_index)

    if total_list_max == 2:
        print('정확한 측정이 어렵습니다. 분석 이미지를 바꿔주세요')
        listbox.insert(5, '정확한 측정이 어렵습니다. 분석 이미지를 바꿔주세요')
        listbox.insert(6, '======================================================================================')
    else:
        print('결과는',(total_list_index+1),'입니다')
        listbox.insert(5, '해당 이미지는 ' + str(total_list_index+1) + ' 단계 입니다.')
        listbox.insert(6, '======================================================================================')
    #표시하는 부분
    userimg = Label(userimg_frame, image=logo)
    p_var2.set(100)
    progress_bar2.update()


def make_check_img():
    makepage = tkinter.Toplevel()
    global filenames
    filenames=[]
    def choose_files():
        list_file.delete(0, END)
        global filenames
        filenames = filedialog.askopenfilenames(title="이미지들을 선택해주세요")
        for filename in filenames:
            list_file.insert(0,filename)
    def make_pallete():
        hists=[]
        colorSorts=[]
        k=10
        clt = KMeans(n_clusters = k)
        for i,f in enumerate(filenames):
            #print('파일이름',f)
            image = image_Reshape(f)
            #print(image)
            clt.fit(image)
            hist = centroid_histogram(clt)
            hist = hist_sort(hist)
            hists.append(hist)
            colorSort = clt.cluster_centers_
            colorSorts.append(colorSort)

            progress=(i+1)/len(filenames)*100
            p_var.set(progress)
            progress_bar.update()

        #print(hists)
        #print(colorSorts)
        barsum1 = np.zeros((50, 300, 3), dtype=np.uint8)
        for i in range(len(hists)):
            bar = plot_colors2(colorSorts[i])
            barsum1 = np.concatenate((barsum1, bar), axis=0)

        # 공백제거
        barsum1 = rebar(barsum1)
        #print(barsum1)
        #plt.figure()
        #plt.axis("off")
        #plt.imshow(barsum1)
        #plt.show()
        #print(barsum1.shape)
        #print(type(barsum1))
        #barsum1_img = Image.fromarray(barsum1)  # 넘파이를 이미지로 바꿔줌
        #print(type(barsum1_img))
        # bar 표시

        bhists = []  # 색비율
        bcolorSorts = []  # RGB값
        #
        k = 30  # 색상 K개 추출
        b_clt = KMeans(n_clusters=k)
        barsum1 = barsum1.reshape((barsum1.shape[0] * barsum1.shape[1], 3))
        b_clt.fit(barsum1)
        hist = centroid_histogram(b_clt)
        hist = hist_sort(hist)
        bhists.append(hist)
        colorSort = b_clt.cluster_centers_
        bcolorSorts.append(colorSort)
        #
        for i in range(len(bhists)):
            fianl_bar = plot_colors(bhists[i], bcolorSorts[i])
        #    plt.figure()
        #     plt.axis("off")
        #     plt.imshow(bar)
        #    plt.show()
        #
        global final_bar
        final_bar = Image.fromarray(fianl_bar)
        print(final_bar)
        list_file.insert(0, '완료 되었습니다! 저장하기를 눌러주세요')

    def save_pallete():
        print(final_bar)
        result = filedialog.asksaveasfilename(initialdir="/", title="Select file", defaultextension='.png',
                                              filetypes=([('PNG 파일', '*.png')]))
        if result:
            print(result)
            final_bar.save(result)


    file_frame=Frame(makepage)
    file_frame.pack()
    choose_btn=Button(file_frame, text="이미지 선택하기",padx=5, pady=5,width=25,command=choose_files)
    choose_btn.pack(side='left')
    make_btn = Button(file_frame, text="분석용 이미지 만들기",padx=5, pady=5,width=25, command=make_pallete)
    make_btn.pack(side='left')
    make_btn = Button(file_frame, text="분석용 이미지 저장", padx=5, pady=5, width=25, command=save_pallete)
    make_btn.pack(side='right')

    list_frame = Frame(makepage)
    list_frame.pack(fill="both")

    scrollbar = Scrollbar(list_frame)
    scrollbar.pack(side="right",fill="y")
    list_file = Listbox(list_frame,selectmode="extended", height=15,yscrollcommand=scrollbar.set)
    list_file.pack(side="left",fill="both",expand=True)
    scrollbar.config(command=list_file.yview)

    frame_progress = LabelFrame(makepage,text="진행상황")
    frame_progress.pack(fill="x")
    p_var = DoubleVar()
    progress_bar = ttk.Progressbar(frame_progress,maximum=100,variable=p_var)
    progress_bar.pack(fill="x")
    makepage.title("이미지 자르기")




userimgload_btn  = Button(btn_frame, text="이미지 불러오기",bg='#575757',font=2,fg='white',width=40,height=3,command= userimgload)#lambda:[userimgload(), checkliver()]여러개의 함수를 입력하기 위해선 labmda로 구현 해야함
userimgload_btn.pack()
userimgcheck_btn = Button(btn_frame, text="검사하기",bg='#575757',fg='white',font=2,width=40,height=3, command=checkliver)
userimgcheck_btn.pack()


userimgcut_btn = Button(btn_frame, text="이미지 자르기",bg='#575757',fg='white',font=2,width=40,height=3,command=file_ok)
userimgcut_btn.pack()
userimgcut_btn = Button(btn_frame, text="분석용 이미지 만들기",bg='#575757',fg='white',font=2,width=40,height=3,command = make_check_img)
userimgcut_btn.pack()

#분석 가져오기
lv1load_btn = Button(lv1_frame, text="1단계 불러오기",bg='#575757',fg='white',width=13,padx=10,command=lv1load)
lv1load_btn.pack(side='right',padx=(0,40))
lv1.pack()
lv2load_btn = Button(lv2_frame, text="2단계 불러오기",bg='#575757',fg='white',width=13, command=lv2load)
lv2load_btn.pack(side='right',padx=(0,40))
lv2.pack()
lv3load_btn = Button(lv3_frame, text="3단계 불러오기",bg='#575757',fg='white',width=13, command=lv3load)
lv3load_btn.pack(side='right',padx=(0,40))
lv3.pack()
lv4load_btn = Button(lv4_frame, text="4단계 불러오기",bg='#575757',fg='white',width=13, command=lv4load)
lv4load_btn.pack(side='right',padx=(0,40))
lv4.pack()

listbox_frame =Frame(root_ori,pady=2)
listbox_frame.pack()
listbox=tkinter.Listbox(listbox_frame,width=75, height=7)
#리스트 박스 배치
listbox.pack()

progrss_frame =Frame(root_ori)
p_var2 = DoubleVar()
progress_bar2 = ttk.Progressbar(progrss_frame,maximum=100,variable=p_var2)
progress_bar2.pack(fill="x")
progrss_frame.pack(fill="x",anchor='s',side='bottom')


listbox_frame['bg'] = '#242424'
btn_frame['bg'] = '#242424'
lv1_frame['bg'] = '#242424'
lv2_frame['bg'] = '#242424'
lv3_frame['bg'] = '#242424'
lv4_frame['bg'] = '#242424'
lv1_img_frame['bg'] = '#242424'
lv2_img_frame['bg'] = '#242424'
lv3_img_frame['bg'] = '#242424'
lv4_img_frame['bg'] = '#242424'
progrss_frame['bg'] = '#242424'
userimg_frame['bg'] = '#242424'
userimg_frame_img['bg'] = '#242424'
root['bg'] = '#242424'
root_ori['bg'] = '#242424'
root_ori.mainloop()
