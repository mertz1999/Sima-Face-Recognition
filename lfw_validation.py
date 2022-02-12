

# --- define Functions
def face_detect(file_name):
  trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
  flag = True
  # Choose an image to detect faces in
  img = cv2.imread(file_name)
  return [img], flag


def lfw_validation(model, l2_dist, valid_thresh, lfw_pairs_path='lfw_pairs_path.npy'):
    # --- Define some variables
    lfw_pairs_path = np.load(lfw_pairs_path, allow_pickle=True)
    pairs_dist_list_mat = []
    pairs_dist_list_unmat = []
    tot_len = len(lfw_pairs_path)

    model.eval() # use model in evaluation mode
    with torch.no_grad():
        true_match = 0
        for path in lfw_pairs_path:
            # --- extracting
            pair_one_path = path['pair_one']
            # print(pair_one_path)
            pair_two_path = path['pair_two']
            # print(pair_two_path)
            matched       = int(path['matched'])

            # --- detect face and resize it
            pair_one_img, flag_one = face_detect(pair_one_path)
            pair_two_img, flag_two = face_detect(pair_two_path)

            if (flag_one==False) or (flag_two==False):
                tot_len = tot_len-1
                continue
            

            # --- Model Predict
            pair_one_img = transform_list(pair_one_img[0])
            pair_two_img = transform_list(pair_two_img[0])
            pair_one_embed = model(torch.unsqueeze(pair_one_img, 0).to(device))
            pair_two_embed = model(torch.unsqueeze(pair_two_img, 0).to(device))
       
            # --- find Distance
            pairs_dist = l2_dist.forward(pair_one_embed, pair_two_embed)
            if matched == 1: pairs_dist_list_mat.append(pairs_dist.item())
            if matched == 0: pairs_dist_list_unmat.append(pairs_dist.item())
            

            # --- thrsholding
            if (matched==1 and pairs_dist.item() <= valid_thresh) or (matched==0 and pairs_dist.item() > valid_thresh):
                true_match += 1

    valid_thresh = (np.percentile(pairs_dist_list_unmat,25) + np.percentile(pairs_dist_list_mat,75)) /2
    print("Thresh :", valid_thresh)
    return ((true_match/tot_len)*100, valid_thresh)


