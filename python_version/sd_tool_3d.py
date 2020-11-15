import plotly.graph_objects as go  # For graphics
import numpy as np 

import scipy.io
from scipy import signal
from scipy.signal import convolve2d
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import matplotlib.pyplot as plt
import scipy 


#>>>>>>>>>>>>>>>>> For graphics <<<<<<<<<<<<<<<<<<<<<

def graph_3d(X,Y,Z,title='Title'):
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='RdBu_r',
                                     cmin=np.min(Z),cmax=np.max(Z),
                                     contours = {
                                        "x": {"show": True, "start": 0, "end": 200, "size": 1, "color":"black"},
                                        "y": {"show": True, "start": 0, "end": 200, "size": 1, "color" :"black"}
                                    })])
    fig.update_layout(title=title, autosize=True,
                  width=1000, height=800,
                 scene = {
                    "xaxis": {"nticks": 20},
                    "zaxis": {"nticks": 20},
                    'camera_eye': {"x": 0, "y": -1, "z": 0.2},
                    "aspectratio": {"x": 1, "y": 1, "z": 0.2}
                })
    fig.show()
    
def graph_3d_peaks(X,Y,Z,x,y,z,title='Title'):
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='RdBu_r',
                                     cmin=np.min(Z),cmax=np.max(Z),
                                     contours = {
                                        "x": {"show": True, "start": 0, "end": 200, "size": 1, "color":"black"},
                                        "y": {"show": True, "start": 0, "end": 200, "size": 1, "color" :"black"}
                                    }),
                             go.Scatter3d(x=x,y=y,z=z,
                                            mode='markers',
                                            marker=dict(
                                                size=4,
                                                opacity=1,
                                                symbol='diamond',
                                                color ="yellow"
                                            )
                                        )
                                        ])

    fig.update_layout(title=title, autosize=False,
                  width=1000, height=800,
                 scene = {
                    "xaxis": {"nticks": 20},
                    "zaxis": {"nticks": 20},
                    'camera_eye': {"x": 0, "y": -1, "z": 0.2},
                    "aspectratio": {"x": 1, "y": 1, "z": 0.2}
                })
    fig.show()
    
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>For Mathematical analysis<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2).astype(np.float32)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h.astype(np.float32)

def mmin(X,v):
    X[X>=v]=v
    return X

def mmax(X,v):
    X[X<=v]=v
    return X

def read_sdf(filepath):
    '''Read files with sdf extension.
    
    Inputs:
    - path : str :Path to sdf file
    
    Outputs:
    - X, Y, Z : float : Matrices necessary to graph the data contained 
    in the file with extension .sdf
    
    '''
    #Reading the file
    with open(filepath,"r") as f:
        data = f.readlines()

    # split the data
    ind = data.index("*\n")
    
    #info
    dinfo = data[:ind]
    info = {d.split("=")[0].strip().lower():d.split("=")[-1].strip() for d in dinfo[1:]}

    #data
    M = [float(d.strip()) for d in data[ind+1:-1]]
  
    #For the grid
    npoints = int(info["numpoints"])
    nprofiles = int(info["numprofiles"])
    
    # Get the X, Y, Z values
    [X,Y] = np.meshgrid(range(1,npoints+1),range(1,nprofiles+1))
    Z = np.reshape(M,(int(info["numpoints"]),int(info["numprofiles"])))
    
    # rescale
    X = X * float(info["xscale"]);
    Y = Y * float(info["yscale"]);
    Z = Z * float(info["zscale"]);
    
    return X,Y,Z


def fast_peak_find(d=[], thresh = None,alpha=0.7, filt=None, edg = 3, res = 1, fid = None):
    '''
    inputs:

    - d: The 2D data raw image.
    - thresh: A number between 0 and max(raw_image) to remove background
    - filt: A filter matrix used to smooth the image. The filter size should correspond the charactistic
    size of the peaks.
    -edg: A number > 1 for skipping the first few and the last few 'edge' pixels
    -res: A handle that switches between two peak finding methods:
        1 the local maxima method (default)
        2 the weighted centroid sub-pixel resolution method.

    -fid: (path.mat) In case the user would like to save the peak positions to a file
    
    Outputs:
        detected_peaks: logical matrix with dimension equal to the image, in the position where 
        there is a peak the value will be True, otherwise it will be False.
    '''
    if len(d)==0:
        filt = matlab_style_gauss2D(shape=(15,15),sigma=3) 
        d = conv2(1*(np.random.rand(1024,1024)>0.99995),filt,'same')+(2^8)*np.random.rand(1024,1024)
        
    if not thresh:
        thresh = max((np.min(d.max(axis=1)), np.min(d.max(axis=0))))
        
    if not filt:
        filt = matlab_style_gauss2D((7,7),1)
    
    detected_peaks = [] # in case of image is all zeros
    ## Analyze image
    if d.any(): 
        d = scipy.signal.medfilt2d(d,[3,3]).astype(np.float32)

        # apply threshold
        thresh = 0.5*thresh
        d = d * (d > thresh)
        
        if d.any():

            # smooth the image
            d = conv2(d,filt,'same')

            # Apply again threshold (and change if needed according to SNR)
            d = d * (d > alpha*thresh)
            
            if res == 1:
                #Peak find - using the local maxima approach - 1 pixel resolution
                        
                detected_peaks = detect_peaks(d)
            else:
                #find wrightd centroids of processed image, sub-pixel resolution.
                pass
        
    if fid:
        scipy.io.savemat(fid,detected_peaks)
        
    return detected_peaks

def get_features(Zsurf,pts):
    '''Determines the size of each of the shapes (widths and heights).
    
    Description:
    The function internally runs an iterative clustering algorithm to extract all 
    the points of each sahpe starting from the centroid. The algorithm iteratively
    selects point rings from the centroid, computes the average value (corresponding 
    to the average height of the ring) and then computes the distance of the current 
    ring from the previous one. The reference or stop criterion is the distance of
    the first ring with respect to the centroid. If the distance between the current 
    ring and the previous one is less than that reference, then the grouping ends. 
    Then the last ring found corresponds to the width of the current pattern under 
    study. Finally, the width of the shapes is the diameter of that last ring
    and the heights are simply the evaluation of Zsurf in the centroid. 
    
    Inputs:
    
    -Zsurf: M-by-N matrix of Zsurf values that, seen as a surface, contain     
           certain patterns or shapes whose size (width and height) vary ac-  
           cording to a normal distribution and whose distance between center 
           to center (pitch) along of the X and Y axes also follows a normal  
           distribution    
           
    - pts: centroid points of each of the patterns in Zsurf.
    
    Outputs:
    
    -wc: Width of each of the shapes                                           
    -hc: Height of each of the shapes                                          
    -pc
    '''

    #work size 
    s = np.shape(Zsurf)
    
    #get the surface domain
    Y,X = np.meshgrid(range(0,s[1]),range(0,s[0]))
    
    #get cluster numbers
    K = len(pts)
    
    #create matrices to return each feature
    
    wc = [] # pattern widths
    hc = [] # pattern heights
    pc = [] # pattern pitch
    
    count = 0
    #sweep for each centroid
    for xo,yo in pts:
        
        #create a mask for clustering
        mask = (X-xo)*(X-xo) + (Y-yo)*(Y-yo)
        #variables to comparison
        udz = 1 # z distance between rings for clustering
        r = 1  # initial radius size for the rings

        # --- Clustering phase
        while True:
            #get the current mask (increases according to the radius)
            ind = mask == r*r # a ring of radio 'r'
            #print(np.shape(ind))
            #get the values at the ring
            Z = Zsurf[ind]
            dz = Zsurf[xo,yo]/Z.mean(axis=0) - 1

            #for the first radius value
            if r == 1:
                #set the stopping criterio 'ref'
                ref = dz
                udz = dz
                #for else  radius values
            else:
                #distance between previus and current ring 
                udz = dz - pdz

            # set the current distance as previus distance for next iteration
            pdz = dz

            #increase the radio
            r += 1

            #stopping criterion 
            if udz < ref or r == np.max(mask):
                # clustering finalized
                break

        #last ind has the width information
        # get the width of the current cluster (pattern)
        iy, ix = np.nonzero(ind)
        wc.append(np.max([max(ix)-min(ix), max(iy)-min(iy) ])*1.0)

        #get the heigth of the current cluster
        hc.append(np.abs( Zsurf[yo,xo]-np.min(Z)*1.1 ))
        # hc.append( np.abs( Zsurf[yo,xo]-np.min(Zsurf) ) * 1.0 )


        euc = np.sqrt((pts[:,0]-xo)*(pts[:,0]-xo) + (pts[:,1] - yo)*(pts[:,1]-yo))
        euc = np.sort(euc)
        pc.append( ( euc[1] + euc[1] ) * 0.5 * 0.9 )
    return wc,hc,pc

def get_3d_pattern_statistics(Zsurf, pattype=None):
    ''' Description.

        receives an M-by-N matrix of Zsurf values that, seen as a surface, contain   
         certain patterns or shapes whose size (width and height) vary according to   
         a normal distribution and whose distance between center to center (pitch)    
         along of the X and Y axes also follows a normal distribution. Then the func- 
         tion, also knowing as input the type of patterns (pattype, top hemispheres   
         for example) contained in Zsurf, locate these patterns and extract their     
         main statistics. That is, it determines the normal distributions associated  
         with its size and pitch.                                                     

         Inputs:   

           -Zsurf: M-by-N matrix Z values read from an .sdf file with the special     
                   function read_sdf().                                               

          -pattype: string indicating the type of patterns contained in Zsurf (See    
                    table below)                                                      

         Outputs:   

          -polyshape:    Dictionary with:     
                         - First key is a string specifying the type of 3D shape de- 
                           sired (see the table below to know the types of shapes     
                           available).                                                
                         - Second key is a dictionary with W_statistics                         
                         - Third key is similar to the previous, but with H_statistics                                   


          -distparams:   dictionary of four elements                                   
          -imsize:       Dictionary of two elements specifying the size of the 3D out-    
                         put surface, 'img'.                                          

          -features:    dictionary of four keys.                 

          Table: pattype options                                                      
          +------------+-------------+--------------+---------+-----------------+     
          | shape      |  long form  |  short form  |  size   |  description    |     
          +------------+-------------+--------------+---------+-----------------+     
          +------------+-------------+--------------+---------+-----------------+     
          | hemisphere | 'htop'      |  'ht'        | [w, h]  | w: width        |     
          | top        |             |              |         | h: height       |     
          +------------+-------------+--------------+---------+-----------------+     
          | hemisphere | 'hbottom'   |  'hb'        | [w, h]  | w: width        |     
          | bottom     |             |              |         | h: height       |     
          +------------+-------------+--------------+---------+-----------------+ 
    '''

    Z = Zsurf #- np.min(Zsurf)
    
    #work size 
    ly,lx = np.shape(Z)
    
    # ---get the centroids
    
    # function that find peaks over 2D signals
    pks = fast_peak_find(Z,np.median(np.median(Z,axis=0)))
    
    # then the centroids are
    x,y = np.nonzero(pks)
    centers = np.array([[x_,y_] for x_,y_ in zip(x,y) ])
    print('Nun Peaks found: %d' %centers)
    #---get the features from centroids
    
    # function that computes the features
    wc, hc, pc = get_features(Z, centers)
    
    #--- get the statistics
    
    pmean = np.mean(pc)
    pstd = np.std(pc)
    pmin = min(pc)
    pmax = max(pc)
    
    #width
    wmean = np.mean(wc);
    wstd = np.std(wc);
    wmin = min(wc);
    wmax = max(wc);

    #height
    hmean = np.mean(hc);
    hstd = np.std(hc);
    hmin = min(hc)   
    hmax = max(hc) 
    
    polyshape = {
        'pattype':pattype,
        'W_statistics':{
            'wmean':wmean,
            'wstd':wstd,
            'wmin':wmin,
            'wmax':wmax,
        },
        'H_statistics':{
            'hmean':hmean,
            'hstd':hstd,
            'hmin':hmin,
            'hmax':hmax
       }
    }
    distparams={
        'pmean':pmean,
        'pstd':pstd,
        'pmin':pmin,
        'pmax':pmax
    }
    imsize={
        'ly':ly,
        'lx':lx
    }
    features={
        'wc':wc,
        'hc':hc,
        'pc':pc,
        'centers':centers
    }
    
    return polyshape, distparams, imsize, features

def shape_placement_3d_v1(imsize,polyshape,distparams):
    ''' Shape placement 3D function.
             
        Inputs:
                                                                                      
          -imsize:       dictionary of two elements specifying the size of the 3D out-    
                         put surface, 'img'.                                          
                                                                                                                                                                            
          -polyshape:    Dictionary with::      
                         - First key is a string specifying the type of 3D shape de- 
                           sired (see the table below to know the types of shapes     
                           available).                                                
                         - Second key is a dictionary with W_statistics                         
                         - Third key is similar to the previous, but with H_statistics      
                                                                                      
          -distparams:   dictionary of four elements                               
                                         
          Table: polyshape options                                                    
          +------------+-------------+--------------+---------+-----------------+     
          | shape      |  long form  |  short form  |  size   |  description    |     
          +------------+-------------+--------------+---------+-----------------+     
          +------------+-------------+--------------+---------+-----------------+     
          | hemisphere | 'htop'      |  'ht'        | [w, h]  | w: width        |     
          | top        |             |              |         | h: height       |     
          +------------+-------------+--------------+---------+-----------------+     
          | hemisphere | 'hbottom'   |  'hb'        | [w, h]  | w: width        |     
          | bottom     |             |              |         | h: height       |     
          +------------+-------------+--------------+---------+-----------------+     
                                                                                      
          NOTE: Now all the size parameters of the polygonal shapes (w and h)         
                will be defined by the distribution parameters specified in polyshape 
                                                                                                                                             
       ''' 
    

    wsp = []
    hsp = []
    
    # Estimating the number of shapes per dimension
    
    pmean = distparams["pmean"]
    nshapes = np.array([int(np.ceil(imsize['ly']/pmean)), int(np.ceil(imsize['lx']/pmean))])
    # --- Get the pits for all the shapes
    
    # Getting it from a pseudo normally distribution defined by
    #the pmean and pstd
    pmean = distparams['pmean']
    pstd = distparams['pstd']
    pitchs = pstd*np.random.randn(2,*(nshapes**2))+pmean
    pitchs = pitchs[:,0:nshapes[0],0:nshapes[1]]
    if len(distparams)>2:
        pitchs = mmax(mmin(pitchs,distparams['pmax']),distparams['pmin'])
    # ---  Get the size for all the shapes
    # Getting it from a pseudo normally distribution defined by
    #the meansize & stdsize ditribution parameters

    # weights
    wp = list(polyshape['W_statistics'].values())
    wmean = wp[0]
    wstd = wp[1]
    ws = wstd*np.random.randn(*(nshapes**2))+wmean
    #ws = ws[0:nshapes[0],0:nshapes[1]]
    # wsp.insert(0,ws)
    if len(wp)>2:
        ws = mmax(mmin(ws,wp[3]),wp[2])
        # wsp.insert(1,ws)
    
    #heigts
    hp = list(polyshape['H_statistics'].values())
    hmean = hp[0]
    hstd = hp[1]
    hs = hstd*np.random.randn(*(nshapes**2))+hmean
    #hs = hs[:,0:nshapes[0],0:nshapes[1]]
    # hsp.insert(0,hs)
    if len(hp)>2:
        hs = mmax(mmin(hs,hp[3]),hp[2])
        # hsp.insert(1,hs)
        
    # --- Placement radonming the shape at the image im
    
    # Creating the output 3D image as a point cloud
    X,Y = np.meshgrid(range(imsize['ly']),range(imsize['lx']))
    img = np.zeros((3,*np.shape(X)))
    img[0,:,:] = X
    img[1,:,:] = Y
    img[2,:,:] = 0.0
    
    xo = np.zeros((1,nshapes[0])).flatten()
    # xprev = xo
    yo = 0
    
    #Sweep over whole image
    #loop extern to control the sweep at x direction
    
    # py = np.zeros(nshapes)
    # px = np.zeros(nshapes)    
    for j in range(nshapes[1]):
        
        #Variable for counting the shapes at the current column
        #term = 1;
        #ishape = y;

        #loop intern to control the sweep at y direction
        
        for i in range(nshapes[0]):

            # Getting the current dx & dy pitchs
            if i==1:
                yo = pitchs[1,i,j]-pmean*0.5
            else:
                yo = yo+pitchs[1,i,j]

            if j==1:
                xo[i] = pitchs[0,i,j]*0.5
            else:
                xo[i] = xo[i]+pitchs[0,i,j]
            
            # Getting the sizes
            w = ws[i,j]
            h = hs[i,j]

            # Getting the grid domain
            # x direction
            js = int(max(np.floor(xo[i]) - np.floor(w),1))
            je = int(min(np.floor(xo[i]) + np.floor(w),imsize['lx']))
            # y direction
            is_ = int(max(np.floor(yo) - np.floor(w),1))
            ie_ = int(min(np.floor(yo) + np.floor(w),imsize['ly']))

            Xd = X[is_:ie_, js:je]
            Yd = Y[is_:ie_, js:je]

            # Getting the dynaminc size mask
            Zd = getshape(polyshape['pattype'],Xd,Yd,w,h,xo[i],yo)
             
            # Placement the shape at the corresponding layer
            img[0,is_:ie_, js:je] = Xd
            img[1,is_:ie_, js:je] = Yd
            img[2,is_:ie_, js:je] = img[2,is_:ie_, js:je] + Zd
    
    return img, pitchs, ws, hs

def shape_placement_3d_v2(imsize,polyshape,distparams):
    ''' Shape placement 3D function.
             
        Inputs:
                                                                                      
          -imsize:       dictionary of two elements specifying the size of the 3D out-    
                         put surface, 'img'.                                          
                                                                                                                                                                            
          -polyshape:    Dictionary with::      
                         - First key is a string specifying the type of 3D shape de- 
                           sired (see the table below to know the types of shapes     
                           available).                                                
                         - Second key is a dictionary with W_statistics                         
                         - Third key is similar to the previous, but with H_statistics      
                                                                                      
          -distparams:   dictionary of four elements                               
                                         
          Table: polyshape options                                                    
          +------------+-------------+--------------+---------+-----------------+     
          | shape      |  long form  |  short form  |  size   |  description    |     
          +------------+-------------+--------------+---------+-----------------+     
          +------------+-------------+--------------+---------+-----------------+     
          | hemisphere | 'htop'      |  'ht'        | [w, h]  | w: width        |     
          | top        |             |              |         | h: height       |     
          +------------+-------------+--------------+---------+-----------------+     
          | hemisphere | 'hbottom'   |  'hb'        | [w, h]  | w: width        |     
          | bottom     |             |              |         | h: height       |     
          +------------+-------------+--------------+---------+-----------------+     
                                                                                      
          NOTE: Now all the size parameters of the polygonal shapes (w and h)         
                will be defined by the distribution parameters specified in polyshape 
                                                                                                                                             
       ''' 
    

    wsp = []
    hsp = []
    
    # Estimating the number of shapes per dimension
    
    pmean = distparams["pmean"]
    pmean_div2 = pmean/2
    nshapes = np.array([int(np.ceil(imsize['ly']/pmean)), int(np.ceil(imsize['lx']/pmean))])
    # --- Get the pits for all the shapes
    
    # Getting it from a pseudo normally distribution defined by
    #the pmean and pstd
    pmean = distparams['pmean']
    pstd = distparams['pstd']
    pr = np.random.randn(2,*(nshapes))*pstd+pmean
    pitchs = (pstd*0.5)*np.random.randn(2,*(nshapes**2))
    pitchs = pitchs[:,0:nshapes[0],0:nshapes[1]]
    if len(distparams)>2:
        pitchs = mmax(mmin(pitchs,(distparams['pmax']-pmean)*0.5),(distparams['pmin']-pmean)*0.5)
    # ---  Get the size for all the shapes
    # Getting it from a pseudo normally distribution defined by
    #the meansize & stdsize ditribution parameters

    # weights
    wp = list(polyshape['W_statistics'].values())
    wmean = wp[0]
    wstd = wp[1]
    ws = wstd*np.random.randn(*(nshapes**2))+wmean
    #ws = ws[0:nshapes[0],0:nshapes[1]]
    wsp.insert(0,ws)
    if len(wp)>2:
        ws = mmax(mmin(ws,wp[3]),wp[2])
        wsp.insert(1,ws)
    
    #heigts
    hp = list(polyshape['H_statistics'].values())
    hmean = hp[0]
    hstd = hp[1]
    hs = hstd*np.random.randn(*(nshapes**2))+hmean
    #hs = hs[:,0:nshapes[0],0:nshapes[1]]
    hsp.insert(0,hs)
    if len(hp)>2:
        hs = mmax(mmin(hs,hp[3]),hp[2])
        hsp.insert(1,hs)
        
    # --- Placement radonming the shape at the image im
    
    # Creating the output 3D image as a point cloud
    X,Y = np.meshgrid(range(imsize['ly']),range(imsize['lx']))
    img = np.zeros((3,*np.shape(X)))
    img[0,:,:] = X
    img[1,:,:] = Y
    img[2,:,:] = 0.0
    
    xo = np.zeros((1,nshapes[0])).flatten()
    xprev = xo
    yo = 0
    
    #Sweep over whole image
    #loop extern to control the sweep at x direction
    
    py = np.zeros(nshapes)
    px = np.zeros(nshapes)    
    for j in range(nshapes[1]):
        
        #Variable for counting the shapes at the current column
        #term = 1;
        #ishape = y;

        #loop intern to control the sweep at y direction
        
        for i in range(nshapes[0]):
            
            xprev[i] = xo[i]
            yprev = yo
            
            xo[i] =(j-1)*pmean+pmean_div2+pitchs[0,i,j]
            yo = (i-1)*pmean+pmean_div2+pitchs[1,i,j]
            
            if i > 1:
                py[i-1,j] = yo-yprev
            
            if j > 1:
                px[j-1,i] = xo[i]-xprev[i]
                
            #Getting the size
            w = ws[i,j]
            h = hs[i,j]
            
            #Getting the grid domain
            # x direction
            js = int(max(np.floor(xo[i]) - np.floor(w),1))
            je = int(min(np.floor(xo[i]) + np.floor(w),imsize['lx']))
            # ydirection
            is_ = int(max(np.floor(yo) - np.floor(w),1))
            ie_ = int(min(np.floor(yo) + np.floor(w),imsize['ly']))
            #print(f"js : {js} je: {je} is_ : {is_} ie_ :{ie_}")
            Xd = X[is_:ie_, js:je];
            Yd = Y[is_:ie_, js:je];

            # Getting the dynaminc size mask
            Zd = getshape(polyshape['pattype'],Xd,Yd,w,h,xo[i],yo)
            
            # Placement the shape at the corresponding layer
            
            img[0,is_:ie_, js:je] = Xd
            img[1,is_:ie_, js:je] = Yd
            img[2,is_:ie_, js:je] = img[2,is_:ie_, js:je] + Zd
        
    pr[0,:,:] = px;
    pr[1,:,:] = py;
    
    return img, pr, wsp, hsp
    
def getshape(shape,X,Y,w,h,xo,yo):
    
    sh = shape.lower()
    
    if sh == 'ht' or sh == 'htop':
        X,Y,Z = hole_top(X,Y,w,h,xo,yo)
    elif sh == 'hb' or sh == 'hbottom':
        X,Y,Z = hole_top(X,Y,w,h,xo,yo)
        Z = -Z
    elif sh =='pt' or sh == 'cylinder':
        X,Y,Z = hemisphere(X,Y,w,h,xo,yo)
    elif sh == 'pb' or sh == 'cone':
        X,Y,Z = hemisphere(X,Y,w,h,xo,yo)
        
    return Z
        
def hole_top(X,Y,w,h,xo,yo):
    t = 3

    if t==1:
        denf = 0.6
        r = w/2
        Z=h*np.exp(-(((X-xo)*(X-xo)/(denf*(r**2)))+((Y-yo)*(Y-yo)/(denf*(r**2)))))

    if t==2:
        denf = 1.1#3.14
        r = w/2
        r = 1/r
        # print(r*denf)
        R = np.sqrt((X-xo)*(X-xo) + (Y-yo)*(Y-yo))
        # Z = h*np.sin(r*denf*R) / (r*denf*R)
        Z = h*np.sinc(r*denf*R)
        Z[Z<0] = 1.0*np.abs(Z[Z<0])
        # Z = np.abs(Z)

    if t==3:
        denf = 1.1#3.14
        r = w/2
        r = 1/r
        Z = h*np.multiply( np.sinc(r*denf*(X-xo)), np.sinc(r*denf*(Y-yo)) )
        Z[Z<0] = 0.5*(Z[Z<0])
        # Z[Z<0] = 1.0*np.abs(Z[Z<0])

    return X,Y,Z
    


