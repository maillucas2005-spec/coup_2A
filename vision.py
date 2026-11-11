import cv2.aruco as aruco
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageSequence
import math as m
import time
from pathlib import Path
from scipy.spatial.transform import Rotation
import time

class vision:
    def __init__(self,image_width,image_height,image_only=True,default_window_size=(3840,2160)):
        self.image_width=image_width
        self.image_height=image_height
        self.rawImage=None
        self.image_from_cam=None
        self.codesPossibles=aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        self.detParam=aruco.DetectorParameters()
        self.detParam.minMarkerPerimeterRate=0.01
        self.detParam.cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX
        self.det=aruco.ArucoDetector(self.codesPossibles,self.detParam)
        self.image_only=image_only
        if not image_only:

            print("direct capture on")
            self.cap= cv2.VideoCapture(0, cv2.CAP_V4L2)
  
            print(self.cap.read())
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            self.image_from_cam=Image.fromarray(cv2.cvtColor(self.cap.read()[1],cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            cv2.namedWindow("aruco detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("aruco detector",*default_window_size)


    def get_deformation_coefs(self,images_calib:list,grid_shape:tuple,cell_real_dim:float)->tuple:
        assert images_calib != None, "pas d'image de calibration"
        grid_cell_coos=[]
        grid_shape=grid_shape[0]-1,grid_shape[1]-1
        for y in range(grid_shape[1]):
            for x in range(grid_shape[0]): 
                grid_cell_coos.append(np.array([x*cell_real_dim,y*cell_real_dim,0.0]))
        grid_cell_coos=np.array(grid_cell_coos,dtype=np.float32)
        grid_cell_coos_found=[]
        corners_found=[]
        print("begin calibration")
        for im in images_calib:
            imnp=np.array(im)
            imnp=cv2.cvtColor(imnp,cv2.COLOR_RGB2GRAY).astype(np.uint8)
            success,corners=cv2.findChessboardCorners(imnp,grid_shape)
            if success:
                grid_cell_coos_found.append(grid_cell_coos.copy())
                corners_found.append(corners)
        success,K,D,_,_=cv2.calibrateCamera(grid_cell_coos_found,corners_found,(self.image_width,self.image_height),None,None)
        assert success, "erreur de lecture"
        print("calibration sucess")
        return K,D

    type coos_aru=list
    type ids=list
    def get_aruco(self,image:Image.Image)->tuple[coos_aru,ids]:
        if self.image_only == True:
            self.rawImage=image
        else:
            self.image_from_cam=Image.fromarray(self.cap.read()[1])
            cv2.waitKey(1)
            self.rawImage=self.image_from_cam  
        self.imagenp=np.array(self.rawImage) 
        detections,ids,rejected=self.det.detectMarkers(self.imagenp)
        corners=[]
        print(ids)
        for c in detections:
            corners.append(c.reshape((2,4)))
        coos_aru=[]
        for aru in corners:
            coos_aru.append(aru[0].tolist()+aru[1].tolist())
        if not isinstance(ids,np.ndarray):
            return None,None
        return coos_aru, ids.tolist()


    def load_images(self,path:str,images_extentions:str):
        images=[]
        files=Path(path)
        for f in files.iterdir():
            name=f.name
            if name.split(".")[1] in images_extentions:
                images.append(Image.open(path+"/"+name))

        
        return images

class timer_cl:
    def __init__(self,name):
        self.begin=time.time()
        self.name=name

    def get_dt(self):
        dt=time.time()-self.begin
        print(self.name,dt)
        return dt

class tags_infos:
    def __init__(self,vision:vision,update_coef:bool,images:list,dict_aruco_reel_size:dict,aruco_reel_pos:dict,tags_info_memory_max_len:int):
        self.tags_info_memory_max_len=tags_info_memory_max_len
        self.vi=vision
        self.cam_orientation=None
        self.cam_pos=None
        self.cam_offset_mesured=np.array([])
        self.dict_aruco_reel_size=dict_aruco_reel_size
        self.aruco_reel_pos=aruco_reel_pos
        self.output_image_detections=None
        self.dict_tags_infos={}
        self.corners_ref=np.array([[-1/2,1/2,0],[1/2,1/2,0],[1/2,-1/2,0],[-1/2,-1/2,0]])
        self.table_aruco=(20,21,22,23)#arcuo fixes sur la tables
        if update_coef:
            self.K,self.D=vi.get_deformation_coefs(images,dim_damier_calib,cell_real_dim=30)
            self.utilities.sav_coefs(self.K,self.D)
        else:
            self.K,self.D=self.utilities.read_coefs()
            
    def creat_ref_aruco_tab(self,excluded:list):
        new_mat=[]
        for id in self.table_aruco:
            if id not in excluded:
                new_mat.append(self.aruco_reel_pos[id]+self.corners_ref*self.dict_aruco_reel_size[id])
        return np.vstack(new_mat)

    def cam_offset_rot_calculation(self,image:Image.Image):
        imagenp=np.array(image)
        detections,ids,rejected=self.vi.det.detectMarkers(imagenp)
        ids=ids.flatten().tolist()
        #warning pour les aruco de calibration non détecter
        all_tab_aruco_detected=self.utilities.are_all_in_list(ids,self.table_aruco)
        if all_tab_aruco_detected!=True:
            if len(all_tab_aruco_detected)==len(self.table_aruco):
                self.utilities.warning("***************aucun aruco de calibration non détecter /!\\***************\n") 
                return None
            else:
                self.utilities.warning("\n"+"*"*60+f"\naruco de calibration non détecter: {all_tab_aruco_detected}\n"+"*"*60+"\n") 

        reshaped_detection=[None]*4 #4aruco sur la table
        print(ids)
        for id in range(len(detections)):
            if ids[id] in self.table_aruco:
                reshaped_detection[ids[id]-20]=detections[id].reshape((4,2))# -20 pour mettres les ids d'aruco sur [0,3]
        
        reshaped_detection=[M for M in reshaped_detection if M is not None]
        reshaped_detection=np.array(reshaped_detection).reshape((-1,2))
        sucess,rvec,tvec=cv2.solvePnP(self.creat_ref_aruco_tab(all_tab_aruco_detected if all_tab_aruco_detected != True else []).reshape((-1,3)),reshaped_detection,self.K,self.D,flags=cv2.SOLVEPNP_ITERATIVE)
        rot=cv2.Rodrigues(rvec)[0]
        oriantation=Rotation.from_matrix(rot.T).as_euler(seq="xyz",degrees=True)
        self.cam_pos=-rot.T@tvec.ravel()
        self.cam_orientation=self.to_rad_vector(oriantation)
        return -rot.T@tvec.ravel(),oriantation

    
    class utilities:
        def __init__(self):
            pass
        def warning(text:str,condition:bool=False):
            if not condition:
                print("warning :",text)

        def are_all_in_list(source,are_in):
            not_in=[]
            for e in are_in:
                if e not in source:
                    not_in.append(e)
            return True if len(not_in)==0 else not_in
        
        def get_rotation_matrix_matrix(self,rx,ry,rz,offx,offy,offz)->np.array:
            cx=np.cos(rx)
            cy=np.cos(ry)
            cz=np.cos(rz)
            sx=np.sin(rx)
            sy=np.sin(ry)
            sz=np.sin(rz)
            return np.array([[cy*cz,sx*sy*cz-cx*sz,cx*sy*cz+sx*sz,offx],[cy*sz,sx*sy*sz+cx*cz,cx*sy*sz-sx*cz,offy],[-sy,sx*cy,cx*cy,offz],[0,0,0,1]])

        def get_corrected_matrix(self,coo_in_cam:np.array,angles_in_cam:np.array,rx,ry,rz,offx,offy,offz):
            rot_mat=self.get_rotation_matrix_matrix(rx,ry,rz,offx,offy,offz)
            coo_in_cam=np.array(coo_in_cam)
            angles_in_cam=np.array(angles_in_cam)
            coo=rot_mat@(np.append(coo_in_cam,1))
            angle=rot_mat[0:3,0:3]@angles_in_cam
            return coo, angle

        def sav_coefs(K,D):
            np.save("K",K)
            np.save("D",D)

        def read_coefs():
            return np.load("k.npy"),np.load("D.npy")
         
    class aruco_infos_:
        def __init__(self,coo:tuple,orientation:tuple,time:int):
            self.coo=coo
            self.angle=orientation
            self.time=time

    def to_deg(self,angle):
        return angle*180/m.pi

    def to_deg_vector(self,angles):
        return self.to_deg(angles[0]),self.to_deg(angles[1]),self.to_deg(angles[2])

    def to_rad_vector(self,angles):
        return self.to_rad(angles[0]),self.to_rad(angles[1]),self.to_rad(angles[2])
    
    def to_rad(self,angle):
        return angle*m.pi/180
    
    def get_aruco_pos(self,seeked_tags:tuple,image:Image.Image=None)->dict:
        if self.vi.image_only == True:
            self.rawImage=image
        else:
            print("update")
            self.image_from_cam=Image.fromarray(cv2.cvtColor(self.vi.cap.read()[1],cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            self.image_from_cam.save("aru.png")
            self.rawImage=self.image_from_cam  
        self.imagenp=np.array(self.rawImage) 

        detection_timer=timer_cl("detector")
        detections,ids,rejected=self.vi.det.detectMarkers(self.imagenp) 
        if ids is None:
            self.utilities.warning("no detection")
            return self.dict_tags_infos,self.rawImage
        detection_timer.get_dt()

        ids=ids.flatten().tolist()
        self.output_image_detections=self.rawImage.copy()
        temp_dic_info={}

        detection_timer=timer_cl("loop")
        for i in range(len(ids)):
            aruco_id=ids[i]
            if aruco_id in seeked_tags:
                if aruco_id in self.dict_aruco_reel_size.keys():
                    sucess,rot,coos=cv2.solvePnP(self.corners_ref*self.dict_aruco_reel_size[aruco_id],detections[i],self.K,self.D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    self.utilities.warning("getting coos failed",sucess)
                    
                    coo,angle=self.utilities().get_corrected_matrix(coos.ravel(),cv2.Rodrigues(rot)[0],*self.cam_orientation,*self.cam_pos)[:3]
                    current_aruco=self.aruco_infos_(tuple(coo.round().tolist()),round(self.to_deg(np.atan2(angle[1,0],angle[0,0])),1),time.time())
                    if aruco_id not in temp_dic_info.keys():
                        temp_dic_info[aruco_id]=[current_aruco]
                    else:
                        temp_dic_info[aruco_id].append(current_aruco)
                    self.draw_aruco_boxs(list(detections[i])[0],self.output_image_detections)
                    self.draw_text(self.output_image_detections,str(aruco_id)+str(current_aruco.coo[:3])+str(current_aruco.angle),detections[i][0][0],30)
        detection_timer.get_dt()

        detection_timer=timer_cl("di")
        for i in ids:
            if i in temp_dic_info.keys():
                if i in self.dict_tags_infos.keys():
                    self.dict_tags_infos[i].append(temp_dic_info[i])
                else:
                    self.dict_tags_infos[i]=[temp_dic_info[i]]
        detection_timer.get_dt()
        #cv2.imshow("aruco detector",np.array(self.output_image_detections))
        return self.dict_tags_infos,self.output_image_detections

    
    def draw_aruco_boxs(self,coos,im:Image.Image):
        ImageDraw.Draw(im).polygon(coos,outline="red", width=8)

    def draw_text(self,im:Image.Image,text:str,coo:tuple,size:int):
        ImageDraw.Draw(im).text(coo,text,font=ImageFont.load_default(size=size),fill="black")



images_extentions=["png","jpg"]
folder_path_calibration="calibration"
dim_damier_calib=(9,6)
image_only=True
aruco_real_size={20:100,21:100,22:100,23:100,13:40,1:74}
aruco_reel_pos={22:[600,600,0],23:[2400,600,0],20:[600,1400,0],21:[2400,1400,0]}
vi=vision(image_width=3840,image_height=2160,image_only=image_only)
images_calib=vi.load_images(folder_path_calibration,images_extentions)
tag_det=tags_infos(vi,False,images_calib,aruco_real_size,aruco_reel_pos,tags_info_memory_max_len=10)

single_image=True*image_only

if single_image:
    im=Image.open("t1.png")
    tag_det.cam_offset_rot_calculation(im)
    begin=time.time()
    for i in range(4):
        timer=timer_cl("****************main****************")
        d,im=tag_det.get_aruco_pos((1,20,21,22,23),im)
        timer.get_dt()

else:
    im=Image.open("t1.png")
    tag_det.cam_offset_rot_calculation(im)
    begin=time.time()

    
    timer=timer_cl("****************main****************")
    d,im=tag_det.get_aruco_pos((1,20,21,22,23))
    if 1 in d.keys():
        print("1:",d[1])
    else:
        print("na")
    timer.get_dt()
    # gif=Image.open("0001-0115.gif")
    # images_out=[]
    # id_=0
    # for im in ImageSequence.Iterator(gif):
    #     print(id_)
    #     id_+=1
    #     d=tag_det.get_aruco_pos(im.convert("RGB"))[0]
    #     images_out.append(tag_det.output_image_detections.resize((1920,1080)))

    # images_out[0].save("animation.gif",save_all=True,append_images=images_out[1:],duration=100, loop=0 )


