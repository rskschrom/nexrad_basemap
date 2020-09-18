import numpy as np
from keras.models import load_model

# normalization function
def normalize(data, scale=None):
    dims = data.shape
    min_data = np.min(data, axis=1)
    max_data = np.max(data, axis=1)
    zero_ind = (min_data==max_data)

    min_data.shape = (dims[0], 1)
    max_data.shape = (dims[0], 1)

    if scale==None:
        data_norm = (data-min_data)/(max_data-min_data)
        data_norm[zero_ind] = 0.
    else:
        data_norm = (data-min_data)/scale

    return data_norm, [min_data, max_data]

# calculate kdp with autoencoder
#----------------------------------
def calc_kdp(phidp, dr):
    #autoencoder = load_model('kdp_model.h5')
    autoencoder = load_model('kdp_simple.h5', compile=False)
    nrange = 512

    # extend phidp in range is less than nrange
    print(phidp.shape)
    nran_orig = phidp.shape[1]
    if nran_orig<nrange:
        phidp_ext = np.ma.masked_all([phidp.shape[0],nrange])
        phidp_ext[:,:nran_orig] = phidp
        phidp = phidp_ext[:]

    # make phidp positive
    #phidp[phidp<0.] = phidp[phidp<0.]+360.

    # mask phidp
    phidp = np.ma.masked_array(phidp)
    phidp[phidp.mask] = np.nan
    #q02 = np.nanquantile(phidp[:,:50], 0.02, axis=1)
    #q02.shape = (len(q02), 1)
    #q02_2d = np.tile(q02, (1,phidp.shape[1]))

    phidp = np.ma.masked_invalid(phidp)
    #phidp = np.ma.masked_where(phidp<q02_2d, phidp)
    #phidp = phidp-q02_2d+10.
    phidp[phidp.mask] = -180.
    phidp_norm, pscale = normalize(phidp, scale=540.)

    phidp_norm.shape = (phidp_norm.shape[0],phidp_norm.shape[1],1)
    kdp = np.ma.masked_all(phidp_norm.shape[0:2])
    delta = np.ma.masked_all(phidp_norm.shape[0:2])

    # calculate kdp for first range chunk
    output_test = autoencoder.predict(phidp_norm[:,:nrange,:])
    kdp[:,:nrange] = 10.*output_test[:,:nrange,0]
    #delta[:,:nrange] = 10.*output_test[:,:nrange,1]

    # calculate kdp for each range chunk
    nrange_dat = phidp_norm.shape[1]
    flen = 41
    nchunk = int(nrange_dat/(nrange-flen))
    end_ri = nrange

    for i in range(nchunk-1):
        str_ri = end_ri-flen
        end_ri = str_ri+nrange

        output_test = autoencoder.predict(phidp_norm[:,str_ri:end_ri,:])
        kdp[:,str_ri+flen:end_ri] = 10.*output_test[:,flen:nrange,0]
        #delta[:,str_ri+flen:end_ri] = 10.*output_test[:,flen:nrange,1]

    # scale by range gate width
    kdp = kdp/dr
    phidp = phidp_norm[:,:,0]*540.+pscale[0]

    return kdp[:,:nran_orig], delta[:,:nran_orig], phidp[:,:nran_orig]
