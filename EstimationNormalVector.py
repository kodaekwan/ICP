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
    o3d.visualization.draw_geometries( show_list,point_show_normal=True);

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


def calculate_point_normalvector(p1,p2,p3):
    # p1 기준점(reference point)
    # p1, p2, p3 data type [x, y, z]
    vector21 = p2-p1;
    vector31 = p3-p1;
    out_vector = np.cross(vector21,vector31);# cross product
    vector_size = np.linalg.norm(out_vector);# L2 norm
    normal_vector=out_vector/vector_size;# unitize
    return normal_vector;

def calculate_cross_vector(p1,p2,p3):
    #p1 기준점(reference point)
    #p2,p3 외부점
    vector_1 = p2-p1;
    vector_2 = p3-p1;
    out_vector = np.cross(vector_1,vector_2);
    return out_vector;

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



def show_normal_vector(source,radius= 0.1, near_sample_num = 15):
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

            # 평균점 구하기
            cp_near_points = np.mean(near_points,axis=0).reshape((1,3));

            # centroid화   
            X = near_points-cp_near_points
            
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
            
            # SVD를 이용한 V행렬을 변환값으로 
            normal_vector = V_T.T[:3,2];
            
            # 
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
            source_pcd.paint_uniform_color([1.0, 0, 0])

            sample_pcd = o3d.geometry.PointCloud()
            sample_pcd.points = o3d.utility.Vector3dVector(np.asarray(near_points));
            sample_pcd.paint_uniform_color([0.0, 1.0, 0])

            X = np.arange(-radius*0.5,radius*0.5,radius*0.05);
            Y = np.arange(-radius*0.5,radius*0.5,radius*0.05);
            X, Y = np.meshgrid(X, Y);
            Z = np.zeros_like(X);
            np_plate = np.stack([X, Y, Z],axis=-1)


            plate_pcd = o3d.geometry.PointCloud();
            plate_pcd.points = o3d.utility.Vector3dVector(np.asarray(np_plate.reshape(-1,3)));
            plate_pcd.rotate(R=V_T.T, center=(0, 0, 0));
            plate_pcd.translate((src_point), relative=False);
            plate_pcd.paint_uniform_color([0.0, 0, 1.0])

            coord=o3d.geometry.TriangleMesh.create_coordinate_frame(size=radius*0.5);
            coord.rotate(R=V_T.T, center=(0, 0, 0));
            coord.translate((src_point), relative=False);
            pcd_show([source_pcd,sample_pcd,plate_pcd,coord]);


        else:
            normal_vector_list.append(np.zeros((3)));




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


# def simple_normal_vector(source,radius= 0.1, near_sample_num = 15):
#     # radius : 지정된 반경 범위
#     # near_num : 근접점 갯수 

#     #normal vector를 계산하기 위한 최소 포인트 수
#     point_num = 2;

#     # 가까운 거리 search을 효율적으로하기 위하여
#     neigh = NearestNeighbors(n_neighbors=near_sample_num)
#     neigh.fit(source)

#     normal_vector_list = [];
#     for _,src_point in enumerate(source):
#         distances, indices = neigh.kneighbors(src_point.reshape(-1,3), return_distance=True);

#         # flatten
#         distances = distances.ravel()
#         indices = indices.ravel()

#         # 지정된 반경 범위 보다 작은 index만 추출
#         cond = np.where( distances < radius);
        
#         # 지정된 반경내 매칭점 추출
#         indices = indices[cond];
#         distances = distances[cond];
        
#         # 지정된 반경내 매칭점 갯수가 2개 이상일 때(자기자신포함)
#         if(len(indices)>=point_num+1):

#             #조건들을 만족하는 가까운 점(point)들 추출
#             near_points = source[indices];

#             # 3점을 조합의 모든 normal vector 계산
#             center_point = near_points.mean(axis=0);
#             length = len(near_points);
#             temp_nv_list = [];
#             for i_ in range(0,length-1):
#                 for j_ in range(i_+1,length):
#                     nv = calculate_cross_vector(center_point,near_points[i_],near_points[j_])
#                     temp_nv_list.append(nv);
            
#             # 모든 normal vector을 평균화
#             mean_nv = np.stack(temp_nv_list,axis=0).mean(axis=0);

#             # 예외 처리
#             if((len(temp_nv_list)%2==0)and(mean_nv==[0.,0.,0.]).all()):
#                 temp_nv_list.pop(0);
#                 mean_nv = np.stack(temp_nv_list,axis=0).mean(axis=0);

#             # 평균 normal vector를 단위화
#             unit_nv = mean_nv/np.linalg.norm(mean_nv);
#             normal_vector_list.append(unit_nv)
#         else:
#             normal_vector_list.append(np.zeros((3)));
        
#     return np.array(normal_vector_list);


def simple_normal_vector(source):
    #normal vector를 계산하기 위한 최소 포인트 수
    
    near_sample_num = 3
    
    # 가까운 거리 search을 효율적으로하기 위하여
    neigh = NearestNeighbors(n_neighbors=near_sample_num)
    neigh.fit(source)

    normal_vector_list = [];
    for _,src_point in enumerate(source):
        distances, indices = neigh.kneighbors(src_point.reshape(-1,3), return_distance=True);

        # flatten
        distances = distances.ravel();
        indices = indices.ravel();

        p1,p2,p3 = source[indices];
        nv = calculate_point_normalvector(p1,p2,p3);

        normal_vector_list.append(nv);
        
    return np.array(normal_vector_list);

def simple_mean_normal_vector(source,near_sample_num = 60):
    #near_sample_num : normal vector를 계산하기 위한 최소 포인트 수


    # 가까운 거리 search을 효율적으로하기 위하여
    neigh = NearestNeighbors(n_neighbors=near_sample_num)
    neigh.fit(source)

    normal_vector_list = [];
    for _,src_point in enumerate(source):
        distances, indices = neigh.kneighbors(src_point.reshape(-1,3), return_distance=True);

        # flatten
        distances = distances.ravel();
        indices = indices.ravel();

        p_list = source[indices];

        nv_buff = [];
        mean_p = np.mean(p_list,axis=0);
        for idx_ in range(0,len(p_list)-1,1):
            nv = calculate_point_normalvector(mean_p,p_list[idx_],p_list[idx_+1]);
            nv_buff.append(nv);
        
        nv = np.mean(np.stack(nv_buff),axis=0);
        nv = nv/np.linalg.norm(nv);

        normal_vector_list.append(nv);
        
    return np.array(normal_vector_list);



if __name__ == "__main__":
    coord=o3d.geometry.TriangleMesh.create_coordinate_frame();
    theta_sample_num = 100;
    theta = np.arange(0.0,2*np.pi,(2*np.pi/100));
    r = 1.0
    x = r*np.cos(theta);
    y = r*np.sin(theta);
    z = np.zeros_like(x);

    

    #0. generate data 
    target=np.stack([x,y,z],axis=-1);
    source=pcd_rotation(target,45.0,0,0);
    init_source = source.copy();
    #source = init_source+[0.1,0.1,0.1];
    source = init_source;
    pcd_show([source,coord]);
    
    #1. geometry normal vector
    cube = o3d.geometry.TriangleMesh.create_box()
    source = np.asarray(cube.sample_points_poisson_disk(5000).points);

    source_nv = simple_normal_vector(source);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);
    

    #2. 노이즈 평면 데이터 생성
    plate_X = np.arange(-1.0,1.0,0.1);
    plate_Y = np.arange(-1.0,1.0,0.1);
    plate_X, plate_Y = np.meshgrid(plate_X, plate_Y);
    plate_Z = np.zeros_like(plate_X);
    np_plate = np.stack([plate_X,plate_Y,plate_Z],axis=-1);
    source = np_plate.reshape((-1,3)) + np.random.uniform(-0.05,0.05,(400,3));
    pcd_show([source,coord]);

    #3. geometry normal vector
    source_nv = simple_normal_vector(source);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);


    #4. geometry normal vector
    #source_nv = estimation_normal_vector(source,radius=0.5,near_sample_num=60);
    source_nv = simple_mean_normal_vector(source,30)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);

    exit()

    #cube = o3d.geometry.TriangleMesh.create_box()
    cube = o3d.geometry.TriangleMesh.create_cylinder()
    cube_pcd = np.asarray(cube.sample_points_poisson_disk(5000).points);
    show_normal_vector(cube_pcd,radius=0.2,near_sample_num=30)
    exit()

    #원에 대한 법선(normal vector) 구하기
    target_nv = simple_normal_vector(target);
    
    target_with_nv = o3d.geometry.PointCloud()
    target_with_nv.points = o3d.utility.Vector3dVector(np.asarray(target));
    target_with_nv.normals = o3d.utility.Vector3dVector(np.asarray(target_nv));
    print(target_nv)
    pcd_show([target_with_nv]);


    
    

    cube_pcd_nv = simple_normal_vector(cube_pcd);
    target_with_nv = o3d.geometry.PointCloud()
    target_with_nv.points = o3d.utility.Vector3dVector(np.asarray(cube_pcd));
    target_with_nv.normals = o3d.utility.Vector3dVector(np.asarray(cube_pcd_nv));
    pcd_show([target_with_nv]);



    cube_pcd_nv = estimation_normal_vector(cube_pcd);
    target_with_nv = o3d.geometry.PointCloud()
    target_with_nv.points = o3d.utility.Vector3dVector(np.asarray(cube_pcd));
    target_with_nv.normals = o3d.utility.Vector3dVector(np.asarray(cube_pcd_nv));
    pcd_show([target_with_nv]);


    # target_with_nv = o3d.geometry.PointCloud()
    # target_with_nv.points = o3d.utility.Vector3dVector(np.asarray(target));
    # target_with_nv.normals = o3d.utility.Vector3dVector(np.asarray(target_nv));

    # pcd_show([source,target,coord,target_with_nv])



    #5
    # source = init_source+[0.1,0.1,0.1];
    # pcd_show([source,target,coord])
    # error, Tm = ICP(source,target,iteration=30);
    # source = pcd_transform(source,Tm);
    # pcd_show([source,target,coord])
    # print("error: ",error)

