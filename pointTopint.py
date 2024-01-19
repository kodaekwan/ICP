import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

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
    #print(np.matmul(np.matmul(yaw_T,pitch_T),roll_T) )
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

def find_approximation_transform_LS(source, target, init_q = np.array([0,0,0,0,0,0])):
    #    Least Squares using jacobian
    
    #     objective function        =  min Σ|error|
    #     error (e)                 =  R * source + t - target

    #     Δpoint                    =  Jacobian(θ) * [t_x,t_y,t_z,r_x,r_y,r_z]
    #     θ                         =  (r_x, r_y, r_z)
    
    #     source                    =  source + Δsource
    #     Δsource                   ≒  error (linear)

    #     [t_x,t_y,t_z,r_x,r_y,r_z] =  inverse(Jacobian(θ)) * error
    #     inverse(Jacobian(θ)) is inverse(J), you can calculate the inverse(J) using puedoinverse.
    #     puedoinverse(J)           =  J_T * inv(J_T * J_T) (<-if J matrix is Full-Rank )
    #     [t_x,t_y,t_z,r_x,r_y,r_z] =  [J_T * inv(J_T * J_T)] * [J_T * error]
    
    #                |  ∂e      ∂e      ∂e      ∂e     ∂e      ∂e    |
    #     Jacobian = | ----- , ----- , ----- , ----- ,----- , -----  |
    #                | ∂t_x    ∂t_y     ∂t_z   ∂r_x   ∂r_y    ∂r_z   |
    
    # Partial differentiation
    #      ∂e     | 1 |      ∂e     | 0 |     ∂e     | 0 |    ∂e     |   ∂R           |    ∂e     |   ∂R           |    ∂e     |   ∂R          |
    #     ----- = | 0 |     ----- = | 1 |    ----- = | 0 |   ----- = |  ---- * source |   ----- = |  ---- * source |   ----- = |  ---- * source|
    #     ∂t_x    | 0 | ,   ∂t_y    | 0 | ,  ∂t_z    | 1 | , ∂r_x    |  ∂r_x          | , ∂r_y    |  ∂r_y          | , ∂r_z    |  ∂r_z         |

    
    
    # 이동과 회전 초기값 설정
    trans_x,trans_y,trans_z,theta_x,theta_y,theta_z = init_q;

    sx = np.sin(theta_x);
    sy = np.sin(theta_y);
    sz = np.sin(theta_z);

    cx = np.cos(theta_x);
    cy = np.cos(theta_y);
    cz = np.cos(theta_z);

    # 회전 행렬
    R = [   [cz*cy,         cz*sy*sx - sz*cx,       cz*sy*cx + sx*sz],
            [sz*cy,         sz*sy*sx + cz*cx,       sz*sy*cx - cz*sx],
            [-sy,           cy*sx,                  cy*cx]]

    # rx 편미분한 회전 행렬
    dRx = [ [0,             cz*sy*cx + sz*sx,       -cz*sy*sx + cx*sz],
            [0,             sz*sy*cx - cz*sx,       -sz*sy*sx - cz*cx],
            [0,             cy*cx,                  -cy*sx]]
    
    # ry 편미분한 회전 행렬
    dRy = [ [-cz*sy,        cz*cy*sx,               cz*cy*cx],
            [-sz*sy,        sz*cy*sx,               sz*cy*cx],
            [-cy,           -sy*sx,                  -sy*cx]]
    
    # rz 편미분한 회전 행렬
    dRz = [ [-sz*cy,        -sz*sy*sx - cz*cx,      -sz*sy*cx + sx*cz],
            [cz*cy,         cz*sy*sx - sz*cx,       cz*sy*cx + sz*sx],
            [0,             0,                      0]]
    
    dRx = np.array(dRx)
    dRy = np.array(dRy)
    dRz = np.array(dRz)
    R  = np.array(R)
    
    #Error 계산
    error = (np.dot(R,source.T)+np.array([[trans_x,trans_y,trans_z]]).T-target.T).T;
    
    
    Hn_list = [];
    bn_list = [];

    # 자코비안 계산 및 공분산 계산
    for src_point,error_point in zip(source,error):
        Jacobin = np.zeros((3,6));
        Jacobin[:,:3] = np.eye(3);
        Jacobin[:,3] = np.dot(dRx,src_point).T;
        Jacobin[:,4] = np.dot(dRy,src_point).T;
        Jacobin[:,5] = np.dot(dRz,src_point).T;

        Hn = np.dot(Jacobin.T,Jacobin);
        bn = np.dot(Jacobin.T,error_point);

        Hn_list.append(Hn);
        bn_list.append(bn);

    H = np.array(Hn_list).sum(axis=0);
    b = np.array(bn_list).sum(axis=0).reshape((-1,6));
    
    # 이동과 회전 변화량 계산
    solve_q = -np.dot(np.linalg.pinv(H),b.T);
    
    # 이동과 회전 변화량만큼 누적 
    init_q = init_q + (solve_q.T)[0];   

    # 4x4 Transform matrix로 반환
    trans_x,trans_y,trans_z,theta_x,theta_y,theta_z = init_q;
    sx = np.sin(theta_x);
    sy = np.sin(theta_y);
    sz = np.sin(theta_z);
    cx = np.cos(theta_x);
    cy = np.cos(theta_y);
    cz = np.cos(theta_z);
    R = [   [cz*cy,         cz*sy*sx-sz*cx,     cz*sy*cx+sx*sz],
            [sz*cy,         sz*sy*sx+cz*cx,     sz*sy*cx-cz*sx],
            [-sy,           cy*sx,              cy*cx]];
    R  = np.array(R)
    Tm = np.eye(4);
    Tm[:3,:3] = R[:3,:3]
    Tm[:3,3] = np.array([[trans_x,trans_y,trans_z]]);
    return Tm, init_q



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
    

def ICP_LS(source, target, iteration = 10, threshold = 1e-7):
    #    Least Squares using jacobian
    # 초기 자세(Pose) 정의 
    Tm = np.eye(4);
    init_q = np.array([0,0,0,0,0,0])

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
        Tm,init_q = find_approximation_transform_LS(local_source,target[indices],init_q);

        
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

    target = np.stack([x,y,z],axis=-1);
    source = pcd_rotation(target,45.0,0,0)+np.array([0.1,0.1,0.1]);
    init_source = source.copy();

    #1 
    pcd_show([source,target,coord])


    #2
    pcd_show([source,target,coord])
    distances,indices=find_near_point(source,target);
    distances2,indices2=nearest_neighbor(source,target);
    print("near distances :",distances)
    print("near indices :",indices)
    print("near distances diff :",distances-distances2)
    print("near indices diff:",indices-indices2)

    #3
    pcd_show([source,target,coord])
    #distances,indices=find_near_point(source,target);
    distances,indices=nearest_neighbor(source,target);
    Tm = find_approximation_transform(source,target[indices]);
    source = pcd_transform(source,Tm);
    pcd_show([source,target,coord])
    print("#3 near indices :",indices);
    
    #4
    source = init_source+[0.1,0.1,0.1];
    pcd_show([source,target,coord])
    #distances,indices=find_near_point(source,target);
    distances,indices=nearest_neighbor(source,target);
    Tm = find_approximation_transform(source,target[indices]);
    source = pcd_transform(source,Tm);
    pcd_show([source,target,coord])
    print("#4 near indices :",indices);

    #5
    source = init_source+[0.1,0.1,0.1];
    pcd_show([source,target,coord])
    error, Tm = ICP(source,target,iteration=30);
    source = pcd_transform(source,Tm);
    pcd_show([source,target,coord])
    print("error: ",error)

    #6
    source = init_source+[0.1,0.1,0.1];
    pcd_show([source,target,coord])
    error, Tm = ICP_LS(source,target,iteration=30);
    source = pcd_transform(source,Tm);
    pcd_show([source,target,coord])
    print("error: ",error)

    

