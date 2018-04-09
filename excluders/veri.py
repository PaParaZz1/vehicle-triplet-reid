import numpy as np
class Excluder(object):
    def __init__(self, gallery_fids):
        # Store the gallery data
        # self.gallery_fids = map(lambda x:x.split('/')[-1], gallery_fids)
        # self.gallery_fids = [x.split('/')[-1] for x in gallery_fids]
        self.gallery_fids = np.array([x.split('/')[-1] for x in gallery_fids])

    def __call__(self, query_fids):
        # Only make sure we don't match the exact same image.
        # query_fids = map(lambda x:x.split('/')[-1], query_fids)
        # query_fids = [x.split('/')[-1] for x in query_fids]
        query_fids = np.array([x.split('/')[-1] for x in query_fids])
        return self.gallery_fids[None] == query_fids[..., None]
