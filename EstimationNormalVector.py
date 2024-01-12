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
    
    #1. 
    source_nv = simple_normal_vector(source);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);
    

    #2. 
    plate_X = np.arange(-1.0,1.0,0.1);
    plate_Y = np.arange(-1.0,1.0,0.1);
    plate_X, plate_Y = np.meshgrid(plate_X, plate_Y);
    plate_Z = np.zeros_like(plate_X);
    np_plate = np.stack([plate_X,plate_Y,plate_Z],axis=-1);
    source = np_plate.reshape((-1,3)) + np.random.uniform(-0.05,0.05,(400,3));
    pcd_show([source,coord]);

    #3. 
    source_nv = simple_normal_vector(source);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);


    #4. Cube nv estimation 
    cube = o3d.geometry.TriangleMesh.create_box()
    source = np.asarray(cube.sample_points_poisson_disk(5000).points);
    source_nv = simple_normal_vector(source);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);


    #4. cylinder nv estimation 
    cylinder = o3d.geometry.TriangleMesh.create_cylinder()
    source = np.asarray(cylinder.sample_points_poisson_disk(5000).points);
    source_nv = simple_normal_vector(source);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);



    #5. 
    np_plate = np.stack([plate_X,plate_Y,plate_Z],axis=-1);
    source = np_plate.reshape((-1,3)) + np.random.uniform(-0.05,0.05,(400,3));
    source_nv = estimation_normal_vector(source,0.5,60)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);

    #6. Cube nv estimation 
    cube = o3d.geometry.TriangleMesh.create_box()
    source = np.asarray(cube.sample_points_poisson_disk(5000).points);
    source_nv = estimation_normal_vector(source,radius=0.1,near_sample_num=30);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);


    #6. cylinder nv estimation 
    cylinder = o3d.geometry.TriangleMesh.create_cylinder()
    source = np.asarray(cylinder.sample_points_poisson_disk(5000).points);
    source_nv = estimation_normal_vector(source,radius=0.1,near_sample_num=30);
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.asarray(source));
    source_pcd.normals = o3d.utility.Vector3dVector(np.asarray(source_nv));
    pcd_show([source_pcd,coord]);

