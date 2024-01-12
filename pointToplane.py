import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import transformations as transform



def calculate_fitting_normalvector(source):
    # 평균점 구하기
    cp_source = np.mean(source,axis=0).reshape((1,3));

    # centroid화   
    X = source-cp_source
    
    # 공분산행렬 계산
    D = np.dot(X.T,X);

    # 특이값분해
    U,S,V_T = np.linalg.svd(D.T);
    
    # 근사행렬 == 회전행렬 계산
    R = np.dot(V_T,U);

    # reflection case <- SVD 문제 
    if np.linalg.det(R) < 0:
        V_T[2, :] *= -1
        # R = np.dot(V_T,U)
    
    # normal vector 추출
    return V_T.T[:3,2];


def estimation_normal_vector(source,radius= 0.1, near_sample_num = 15):
    # radius : 지정된 반경 범위
    # near_num : 근접점 갯수 

    #normal vector를 계산하기 위한 최소 포인트 수
    point_num = 2;

    # 가까운 거리 search을 효율적으로하기 위하여
    neigh = NearestNeighbors(n_neighbors=near_sample_num)
    neigh.fit(source)

    normal_vector_list = [];
    for _,src_point in enumerate(source):
        distances, indices = neigh.kneighbors(src_point.reshape(-1,3), return_distance=True);

        # flatten
        distances = distances.ravel()
        indices = indices.ravel()

        # 지정된 반경 범위 보다 작은 index만 추출
        cond = np.where( distances < radius);
        
        # 지정된 반경내 매칭점 추출
        indices = indices[cond];
        distances = distances[cond];
        
        # 지정된 반경내 매칭점 갯수가 2개 이상일 때(자기자신포함)
        if(len(indices)>=point_num+1):

            #조건들을 만족하는 가까운 점(point)들 추출
            near_points = source[indices];

            #분산과 평면 fitting 이용한 normal vector 계산
            mean_normal_vector = calculate_fitting_normalvector(near_points);
            normal_vector_list.append(mean_normal_vector/np.linalg.norm(mean_normal_vector))

        else:
            normal_vector_list.append(np.zeros((3)));
        

    return np.array(normal_vector_list);


def pcd_show(point_clouds=[]):
    show_list = [];
    for point_cloud in point_clouds:
        if(type(point_cloud).__module__ == np.__name__):
            np_point_cloud = np.array(point_cloud);
            np_point_cloud = np_point_cloud.reshape((-1,3));
            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(np_point_cloud));
            show_list.append(o3d_point_cloud);
        else:
            show_list.append(point_cloud);
    o3d.visualization.draw_geometries( show_list,point_show_normal=False);

def pcd_rotation(point_cloud,roll_deg=0.0,pitch_deg=0.0,yaw_deg=0.0):
    roll_T = np.array([[1,0,0],
                       [0,np.cos(np.deg2rad(roll_deg)),-np.sin(np.deg2rad(roll_deg))],
                       [0,np.sin(np.deg2rad(roll_deg)),np.cos(np.deg2rad(roll_deg))],
                       ])
    pitch_T = np.array([[np.cos(np.deg2rad(pitch_deg)),0,np.sin(np.deg2rad(pitch_deg))],
                       [0,1,0],
                       [-np.sin(np.deg2rad(pitch_deg)),0,np.cos(np.deg2rad(pitch_deg))],
                       ])
    yaw_T = np.array([[np.cos(np.deg2rad(yaw_deg)),-np.sin(np.deg2rad(yaw_deg)),0],
                       [np.sin(np.deg2rad(yaw_deg)),np.cos(np.deg2rad(yaw_deg)),0],
                       [0,0,1],
                       ])
    np_point_cloud = point_cloud.reshape((-1,3));
    t_pcd = np.matmul(np_point_cloud,np.matmul(np.matmul(yaw_T,pitch_T),roll_T));
    print(np.matmul(np.matmul(yaw_T,pitch_T),roll_T) )
    return t_pcd;

def pcd_transform(point_cloud,Tm):
    np_point_cloud = point_cloud.reshape((-1,3));
    num = np_point_cloud.shape[0];
    np_point_cloud = np.concatenate([ np_point_cloud,np.ones((num,1)) ],axis=-1);
    t_pcd = np.dot(Tm,np_point_cloud.T).T;
    return t_pcd[:,:3];

def nearest_neighbor(sorce, target, n_neighbors=1):
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(target)
    distances, indices = neigh.kneighbors(sorce, return_distance=True)
    return distances.ravel(), indices.ravel()

def find_near_point(sorce, target):
    # 근접점(최소거리)거리와 비교 매칭된 dst index 리스트들
    minmum_dist_list = [];
    minmum_idx_list = [];
    

    # source 점(point) 하나씩 조회
    for idx_s,(src_x,src_y,src_z) in enumerate(sorce): 
        minmum_dist = -1.0;
        minmum_idx = -1.0;

        # target 점(point) 하나씩 조회
        for idx_t,(tg_x,tg_y,tg_z) in enumerate(target):   
            # source와 target point 간 거리계산  
            dist = np.linalg.norm(np.array([tg_x,tg_y,tg_z])-np.array([src_x,src_y,src_z]));

            # 만약 처음이면, 최소거리값과 매칭된 index로 정의
            if(idx_t == 0):
                minmum_dist = dist
                minmum_idx = idx_t;
            else: # 만약 처음이 아니고, 거리가 최소 거리값보다 작으면
                if(minmum_dist>dist):
                    # 최소값 거리값와 매칭 index 업데이트
                    minmum_dist = dist;
                    minmum_idx = idx_t;
        
        # source점과 거리가 가까운 target 점들의 거리와 매칭된 index 기록
        minmum_dist_list.append(minmum_dist);
        minmum_idx_list.append(minmum_idx);
    
    # 근접점(최소거리) 거리와 매칭된 index 반환
    return np.array(minmum_dist_list), np.array(minmum_idx_list)

def find_approximation_transform(sorce, target):
    
    A = target.reshape((-1,3));
    B = sorce.reshape((-1,3));

    # 평균점 구하기
    cp_A = np.mean(A,axis=0).reshape((1,3));
    cp_B = np.mean(B,axis=0).reshape((1,3));

    # centroid화   
    X = A-cp_A
    Y = B-cp_B
    
    # 공분산행렬 계산
    D = np.dot(Y.T,X);

    # 특이값분해
    U,S,V_T = np.linalg.svd(D);
    
    # 근사행렬 == 회전행렬 계산
    R = np.dot(V_T,U);

    # reflection case <- SVD 문제 
    if np.linalg.det(R) < 0:
        V_T[2, :] *= -1
        R = np.dot(V_T,U)

    # 평균점과 구한 회전행렬을 이용하여 이동벡터 계산
    t =  cp_A.T - np.dot(R,cp_B.T)
    
    # 4x4 Transform matrix로 반환
    Tm = np.eye(4);
    Tm[:3,:3] = R[:3,:3]
    Tm[:3,3] = t.T;

    return Tm;





def ICP(source, target, iteration = 10, threshold = 1e-7):
    # 초기 자세(Pose) 정의 
    Tm = np.eye(4);
    # 초기 오차
    Error = 0;

    # 최종 변환 자세
    final_Tm = Tm.copy();

    # 만약 초기 자세를 알면 아래 코드를 활성화
    local_source = pcd_transform(source,Tm)

    # 반복적(Iterative) 계산
    for _ in range(iteration):
        # 가까운 매칭점 계산
        distances, indices=nearest_neighbor(local_source,target);
        
        # 포인트간 평균 거리 계산
        Error = distances.mean()
        
        # 포인트 평균 거리가 한계값보다 작다면 더 이상 찾지 않고 반환
        if(Error<threshold):
            break;

        # 매칭점과 두 포인트클라우드를 이용하여 근사 변환행렬 계산
        Tm = find_approximation_transform(local_source,target[indices]);

        
        # source 포인트클라우드 변환행렬(Tm)을 이용하여 변환
        local_source = pcd_transform(local_source, Tm);

        # 근사된 변환행렬 반복적으로 누적하여 정합된 변환 추정
        final_Tm = np.matmul(Tm,final_Tm);

    return Error,final_Tm;
    

def find_approximation_transform_LS(sorce, target, target_normal):
    # Non-linear Least-squares based ICP

    num = sorce.shape[0];
    normal = target_normal;

    A = np.zeros((num,6));
    b = np.zeros((num,1));
    diff = target-sorce;


    for idx in range(num):

        sx = sorce[idx,0]
        sy = sorce[idx,1]
        sz = sorce[idx,2]

        tx = target[idx,0]
        ty = target[idx,1]
        tz = target[idx,2]

        nx = normal[idx,0]
        ny = normal[idx,1]
        nz = normal[idx,2]

        A[idx,0] = (nz * ty) - (ny * tz)
        A[idx,1] = (nx * tz) - (nz * tx)
        A[idx,2] = (ny * tx) - (nx * ty)
        A[idx,3] = nx
        A[idx,4] = ny
        A[idx,5] = nz

        for j in range(3):
            b[idx,0] = b[idx,0] - np.dot(diff[idx,j],normal[idx,j]);
    
    A_pinv = np.linalg.pinv(A);

    solve_ = np.dot(A_pinv.T,b);

    Tm = np.eye(4);

    sin_a = np.sin(x[0]);
    cos_a = np.cos(x[0]);
    sin_b = np.sin(x[1]);
    cos_b = np.cos(x[1]);
    sin_y = np.sin(x[2]);
    cos_y = np.cos(x[2]);

    Tm[0,0] = cos_y*cos_b;
    Tm[0,1] = -sin_y*cos_a+cos_y*sin_b*sin_a;
    Tm[0,2] = sin_y*sin_a+cos_y*sin_b*cos_a;
    Tm[1,0] = sin_y*cos_b;
    Tm[1,1] = cos_y*cos_a+sin_y*sin_b*sin_a;
    Tm[1,2] = -cos_y*sin_a+sin_y*sin_b*cos_a;
    Tm[2,0] = -sin_b;
    Tm[2,1] = cos_b*sin_a;
    Tm[2,2] = cos_b*cos_a;

    #translation vector
    Tm[0,3] = x[3];
    Tm[1,3] = x[4];
    Tm[2,3] = x[5];

    return Tm;

def ICP_plane(source, target, target_normal, iteration = 10, threshold = 1e-7):
    # 초기 자세(Pose) 정의 
    Tm = np.eye(4);
    # 초기 오차
    Error = 0;

    # 최종 변환 자세
    final_Tm = Tm.copy();

    # 만약 초기 자세를 알면 아래 코드를 활성화
    local_source = pcd_transform(source,Tm)

    # 반복적(Iterative) 계산
    for _ in range(iteration):
        # 가까운 매칭점 계산
        distances, indices=nearest_neighbor(local_source,target);
        
        # 포인트간 평균 거리 계산
        Error = distances.mean()
        
        # 포인트 평균 거리가 한계값보다 작다면 더 이상 찾지 않고 반환
        if(Error<threshold):
            break;

        # 매칭점과 두 포인트클라우드를 이용하여 근사 변환행렬 계산
        Tm = find_approximation_transform_LS(local_source,target[indices],target_normal);

        
        # source 포인트클라우드 변환행렬(Tm)을 이용하여 변환
        local_source = pcd_transform(local_source, Tm);

        # 근사된 변환행렬 반복적으로 누적하여 정합된 변환 추정
        final_Tm = np.matmul(Tm,final_Tm);

    return Error,final_Tm;


if __name__ == "__main__":
    coord=o3d.geometry.TriangleMesh.create_coordinate_frame();
    theta_sample_num = 100;
    theta = np.arange(0.0,2*np.pi,(2*np.pi/100));
    r = 1.0
    x = r*np.cos(theta);
    y = r*np.sin(theta);
    z = np.zeros_like(x);

    target=np.stack([x,y,z],axis=-1);
    source=pcd_rotation(target,45.0,0,0);
    init_source = source.copy();
    
    
    #4
    source = init_source+[0.1,0.1,0.1];
    pcd_show([source,target,coord])
    error, Tm = ICP(source,target,iteration=30);
    source = pcd_transform(source,Tm);
    pcd_show([source,target,coord])
    print("error: ",error)

    #5
    source = init_source+[0.5,0.1,0.1];
    pcd_show([source,target,coord])
    target_normal=estimation_normal_vector(target,0.2,15);
    error, Tm = ICP_plane(source,target,target_normal=target_normal,iteration=30);
    source = pcd_transform(source,Tm);
    pcd_show([source,target,coord])
    print("error: ",error)

