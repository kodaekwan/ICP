import numpy as np
from matplotlib import pyplot as plt


def LeastSquaresMethod(x_dataset,y_dataset):
    An_list = [];
    Bn_list = [];
    for (x_,y_) in zip(x_dataset,y_dataset):    
        An = [x_*x_,x_, 1];
        Bn = [y_];
        An_list.append(An);
        Bn_list.append(Bn);
    A = np.array(An_list);
    B = np.array(Bn_list);

    # A 유사역행렬 계산 
    A_inv = np.dot(np.linalg.pinv(np.dot(A.T,A)),A.T);

    # 함수 factor 계산
    a_hat,b_hat,c_hat = np.dot(A_inv,B).T[0];

    # 피팅된 함수 구현
    y_hat = a_hat*x_dataset*x_dataset + b_hat*x_dataset +c_hat;
    return y_hat,[a_hat,b_hat,c_hat];


def RANSAC(x_dataset,y_dataset,threshold = 5,iter=100):
    random_sample_num = 3;#랜덤 샘플 수
    c_max = 0;# 최대 만족 갯수 
    dataset_length = len(x_dataset);#데이터길이
    best_parm = [0,0,0];# 최고 피팅 모델
    
    #반복 횟수
    for i in range(iter):
        # 랜덤데이터 인덱스 생성
        random_index = np.random.randint(0, dataset_length, size=random_sample_num)

        # 랜덤데이터 추출
        rand_x = x_dataset[random_index];
        rand_y = y_dataset[random_index];

        # 모델 피팅
        y_hat,parm=LeastSquaresMethod(rand_x,rand_y);

        # 피팅된 모델로 y 추정
        y_hat = parm[0]*x_dataset*x_dataset + parm[1]*x_dataset + parm[2];

        # 정답 y 와 추정 y간 에러 추출
        error = np.abs(y_dataset - y_hat);

        # threshold 보다 작은 error 조건들 추출
        cond = np.where(error < threshold);

        # 조건을 만족하는 갯수
        c = len(cond[0]);

        # 조건 만족수가 높은 피팅 모델을 선택
        if(c_max<c):
            best_parm = parm;
            c_max = c;
        
    return best_parm[0]*x_dataset*x_dataset + best_parm[1]*x_dataset + best_parm[2] ,best_parm

if __name__ == "__main__":
#1 노이즈 데이터 생성
    a,b,c = -1.0,1.0,0.0

    # 2차 함수 데이터 생성
    x = np.arange(-5,10,0.1,dtype=np.float32);
    y = a*x*x +b*x + c;

    # 노이즈 추가
    y += np.random.uniform(-5.0,5.0,size=y.size);

    #생성 데이터 
    plt.plot(x,y);
    plt.show()

#2 노이즈 데이터 피팅
    plt.clf()  # 플롯 초기화
    
    # LSM 피팅
    y_hat,_ = LeastSquaresMethod(x,y)
    
    # 생성 데이터 
    plt.plot(x,y);
    # 피팅 데이터 
    plt.plot(x,y_hat,);
    plt.show()



#2 아웃라이어 추가
    plt.clf()  # 플롯 초기화
    # outlier 생성
    y[50:80] += -30*x[50:80]+0;

    # 생성 데이터 
    plt.plot(x,y);
    plt.show()

#3 아웃라이어 노이즈 데이터 피팅
    plt.clf()  # 플롯 초기화
    
    # LSM 피팅
    y_hat,_ = LeastSquaresMethod(x,y)
    
    # 생성 데이터  
    plt.plot(x,y);
    # 피팅 데이터 
    plt.plot(x,y_hat,);
    plt.show()

    
#4 RANSAC 이용한 피팅
    plt.clf()  # 플롯 초기화
    # RANSAC-LSM 피팅
    y_hat,_ = RANSAC(x,y,threshold=5,iter=100)

    # 현실 데이터 
    plt.plot(x,y);

    # 피팅 데이터 
    plt.plot(x,y_hat,);
    plt.show()
