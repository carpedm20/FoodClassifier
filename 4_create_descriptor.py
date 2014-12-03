from multiprocessing import Pool

from utils import get_train_test

TEST = True 
SAVE = False
LOWE = True

descriptor = "SIFT"

train_images, train_labels, test_images, test_labels = get_train_test(TEST)

pool = Pool(cv2.getNumberOfCPUs()-2)

if descriptor == "SIFT":
    if LOWE:
        print " [!] Lowe's SIFT"
        train_sift_with_null = pool.map(get_sift_lowe, train_images)
        test_sift_with_null = pool.map(get_sift_lowe, test_images)
    else:
        print " [!] OpenCV2's SIFT"
        train_sift_with_null = pool.map(get_sift, train_images)
        test_sift_with_null = pool.map(get_sift, test_images)

    pool.terminate()                                                                                        

    train_sift = removing_null(train_sift_with_null, train_labels)
    reduced_train_sift = np.concatenate(train_sift, axis = 0)

    test_sift = removing_null(test_sift_with_null, test_labels)
    reduced_test_sift = np.concatenate(test_sift, axis = 0)

    nfeatures = reduced_train_sift.shape[0]
    k = int(sqrt(nfeatures))

    kmeans = MiniBatchKMeans(n_clusters = k,
                             init       ='k-means++',
                             n_init     = 10,
                             max_iter   = 100,
                             init_size  = 1000,
                             batch_size = 1000)

    kmeans.fit(reduced_train_sift)

    train_predicted = kmeans.predict(reduced_train_sift)
    test_predicted = kmeans.predict(reduced_test_sift)

    train_hist_features = get_histogram(k, train_sift, train_predicted)
    test_hist_features = get_histogram(k, test_sift, test_predicted)
elif :
