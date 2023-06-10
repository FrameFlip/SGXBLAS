#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdbool>
#include <vector>
#include <string>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <dirent.h>
#include <vector>
#include <algorithm>
#include <filesystem>

#include "mydataloader.h"
#include "utils.h"

using namespace std;
using namespace cv;

using std::vector;
using std::map;

namespace fs = std::filesystem;
string get_stem(const fs::path &p) { return p.stem().string(); }


struct instance
{
    string path;
    int label;
};

map<string, int> find_classes(string directory){
    
    int idx = 0;
    vector<string> classes;
    for (const auto & entry : fs::directory_iterator(directory)){
        // cout << get_stem(entry.path()) << endl;
        string sub_dir = get_stem(entry.path());
        classes.push_back(sub_dir);
    }
    sort(classes.begin(), classes.end());

    // cout<< "check sorted: " << endl;
    // for (auto i : classes)
        // cout << i << endl;

    map<string, int> class_to_idx;

    for (auto i : classes)
        class_to_idx[i] = idx++;

    return class_to_idx;
}

vector<instance> make_dataset(string root){
    
    vector<instance> dataset;

    map<string, int> class_to_idx = find_classes(root);
    
    for (const auto & entry : std::filesystem::directory_iterator(root)){
        string sub_dir = entry.path();
        string class_id = get_stem(entry.path());
        for (const auto & file : std::filesystem::directory_iterator(sub_dir)){
            instance tmp = {file.path(),  class_to_idx[class_id]};
            dataset.push_back(tmp);
        }
    }

    return dataset;
}

bool check_file_existence(const string& filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

void save_dump_data(string dump_root, int set_idx, const vector<int> image_shape, int load_gra, vector<instance> dataset)
{
        const string image_dump_path = dump_root + "images_" + std::to_string(set_idx) + ".dmp";
        const string label_dump_path = dump_root + "labels_" + std::to_string(set_idx) + ".dmp";
        
        const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
        double *image_arr = new double[(load_gra) * total_elems];
        int *label_arr = new int[load_gra];
        
        double mean[3] = {0.485, 0.456, 0.406};
        double std[3] = {0.229, 0.224, 0.225};
        // cout << "check images: " << check_file_existence(image_dump_path) << endl; // 0 for not existed!
    if (not check_file_existence(image_dump_path) or not check_file_existence(label_dump_path))
    {
        Mat original_img, resized_img;
        int target_height = image_shape[1], target_width = image_shape[2];

        printf("Loading sub_dataset: %d  ...\n", set_idx);
        for (int image_cnt = 0; image_cnt < load_gra; ++ image_cnt)
        {

            original_img = imread(dataset[image_cnt].path); // BGR image 

            cv::Mat channels[3];
            cv::split(original_img, channels);
            cv::Mat temp = channels[0];
            channels[0] = channels[2];
            channels[2] = temp;
            cv::Mat output;
            cv::merge(channels, 3, output);

            resize(output, resized_img, Size(256, 256), INTER_LINEAR);
            int crop_top = int( (256 - target_height) / 2.0 );
            int crop_left = int( (256 - target_width) / 2.0 );
            Mat cropped_image = resized_img(Range(crop_top, crop_top+target_height), Range(crop_left, crop_left+target_width));
            // cout << "Cropped size: " << cropped_image.size() << endl;

            // just copy the resized image into the storage
            int idx = 0;
            for (int c = 0; c < image_shape[0]; ++ c)
            {
                for (int i = 0; i < image_shape[1]; ++ i)
                {
                    for (int j = 0; j < image_shape[2]; ++ j)
                    {
                        double tmp = ((unsigned int)cropped_image.at<Vec3b>(i, j)[c]) / 255.0;
                        image_arr[image_cnt * total_elems + (idx ++)] = (tmp - mean[c]) / std[c];
                    }
                }
            }
            label_arr[image_cnt] = dataset[image_cnt].label;

            // if (image_cnt % 1000 == 0)
            // {
            //     if (DEBUG_FLAG) printf("[%d / %d] finished\n", image_cnt, n_images);
            // }
        }

        // dump the loaded data
        printf("Saving loaded sub_dataset: %d  ...\n", set_idx);
        cout << "check save path: " << image_dump_path << endl;
        ofstream out_images(image_dump_path, ios::out | ios::binary);
        ofstream out_labels(label_dump_path, ios::out | ios::binary);

        out_images.write((const char *)image_arr, load_gra * total_elems * sizeof(double) / sizeof(char));
        out_labels.write((const char *)label_arr, load_gra * sizeof(int) / sizeof(char));

        out_images.close();
        out_labels.close();

        delete [] image_arr;
        delete [] label_arr;

        printf("Done sub_dataset: %d !\n", set_idx);
    }
    else
    {
        PRINT_DEBUG("Subdataset existed!\n");
        // printf("Loading saved sub_dataset: %d  ...\n", set_idx);
        // ifstream in_images(image_dump_path, ios::in | ios::binary);
        // ifstream in_labels(label_dump_path, ios::in | ios::binary);

        // in_images.read((char *)image_arr, load_gra * total_elems * sizeof(uchar) / sizeof(char));
        // in_labels.read((char *)label_arr, load_gra * sizeof(int) / sizeof(char));

        // in_images.close();
        // in_labels.close();
        // printf("Done sub_dataset: %d !\n", set_idx);
    }
    // image_arr_set.push_back(image_arr);
    // label_arr_set.push_back(label_arr);


}


TrafficTestLoader::TrafficTestLoader(string root, int batch_size, string dump_root, bool force_reload):
    n_images(12630), mbatch_size(batch_size), 
    image_arr(NULL), label_arr(NULL), image_shape({3, 32, 32}),
    batch_idx(0)
{
    const string image_dump_path = dump_root + "images.dmp";
    const string label_dump_path = dump_root + "labels.dmp";

    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    image_arr = new double[(n_images) * total_elems];
    label_arr = new int[n_images];
    double mean[3] = {0.485, 0.456, 0.406};
    double std[3] = {0.229, 0.224, 0.225};

    if (force_reload or not check_file_existence(image_dump_path) or not check_file_existence(label_dump_path))
    {
        ifstream anno_in;
        char image_filename[10];        // store the image name
        int width, height, roi_x1, roi_y1, roi_x2, roi_y2;
        int classid;
        Mat original_img, resized_img;
        int target_height = image_shape[1], target_width = image_shape[2];
        int label_arr_idx = 0;
        
        vector<string> imgs_paths; 

        struct dirent *sub_dir = nullptr;
        DIR *dp = nullptr;
        dp = opendir(root.c_str());
        if (dp != nullptr) {
            while ((sub_dir = readdir(dp))){
		        if (((string)sub_dir->d_name) == ".." || ((string) sub_dir->d_name) == ".")
        			continue;
                // printf ("subdir: %s\n", sub_dir->d_name);
		        string subdir_path = root + "/" + ((string) sub_dir->d_name);
                // printf ("subdir path: %s\n", subdir_path.c_str());
		
		        DIR *sub_dp = nullptr;
		        struct dirent *filename = nullptr;
		        sub_dp = opendir(subdir_path.c_str());
		        if(sub_dp != nullptr){
			        while((filename = readdir(sub_dp))){
				        if (((string)filename->d_name) == ".." || ((string) filename->d_name) == ".")
					        continue;
				        string file_path = subdir_path + "/" + ((string) filename->d_name);
				        // printf("filename: %s\n", file_path.c_str());
				        imgs_paths.push_back(file_path);
                        string file_name_str = ((string)(sub_dir->d_name)).c_str();
                        int filename_str_len = file_name_str.size();
                        char first_three_str[4] = {file_name_str[filename_str_len-3], file_name_str[filename_str_len-2], file_name_str[filename_str_len-1], '\0'};
                        int label_tmp = atoi(first_three_str);
                        // printf("check readed labels: %d\n", label_tmp);
                        label_arr[label_arr_idx++] = label_tmp;
			        }	
		        }
		        closedir(sub_dp);
  	        }
        }
        closedir(dp);
        cout << "number of image paths: " << imgs_paths.size() <<endl;

        PRINT_DEBUG("Loading images ...\n");
        for (int image_cnt = 0; image_cnt < n_images; ++ image_cnt)
        {
            // printf("check cv2 input path: %s\n", imgs_paths[image_cnt].c_str());
            original_img = imread(imgs_paths[image_cnt]); // BGR image 
            // printf("check loaded images: %d\n", original_img.at<Vec3b>(0, 0)[0]);
            cv::Mat channels[3];
            cv::split(original_img, channels);
            cv::Mat temp = channels[0];
            channels[0] = channels[2];
            channels[2] = temp;
            cv::Mat output;
            cv::merge(channels, 3, output);

            resize(output, resized_img, Size(target_width, target_height), INTER_LINEAR);
            // printf("check loaded images: %d\n", resized_img.at<Vec3b>(0, 0)[0]);
            // just copy the resized image into the storage
            int idx = 0;
            for (int c = 0; c < image_shape[0]; ++ c)
            {
                for (int i = 0; i < image_shape[1]; ++ i)
                {
                    for (int j = 0; j < image_shape[2]; ++ j)
                    {
        		      	double tmp = ((unsigned int)resized_img.at<Vec3b>(i, j)[c]) / 255.0;     
                        image_arr[image_cnt * total_elems + (idx ++)] = (tmp - mean[c]) / std[c];
                    }
                }
            }

            if (image_cnt % 1000 == 0)
            {
                if (DEBUG_FLAG) printf("[%d / %d] finished\n", image_cnt, n_images);
            }
        }

        anno_in.close();

        // dump the loaded data
        PRINT_DEBUG("Saving loaded data ... ");
        ofstream out_images(image_dump_path, ios::out | ios::binary);
        ofstream out_labels(label_dump_path, ios::out | ios::binary);

        out_images.write((const char *)image_arr, n_images * total_elems * sizeof(double) / sizeof(char));
        out_labels.write((const char *)label_arr, n_images * sizeof(int) / sizeof(char));

        out_images.close();
        out_labels.close();

        PRINT_DEBUG("Done!\n");
    }
    else
    {
        PRINT_DEBUG("Loading saved data ... ");
        ifstream in_images(image_dump_path, ios::in | ios::binary);
        ifstream in_labels(label_dump_path, ios::in | ios::binary);

        in_images.read((char *)image_arr, n_images * total_elems * sizeof(double) / sizeof(char));
        in_labels.read((char *)label_arr, n_images * sizeof(int) / sizeof(char));

        in_images.close();
        in_labels.close();

        PRINT_DEBUG("Done!\n");
    }

    // cout << (int)image_arr[10086] << "\t" << label_arr[10086] << endl; // should be: 139  7
}


TrafficTestLoader::~TrafficTestLoader()
{
    if (image_arr != NULL)
        delete [] image_arr;
    if (label_arr != NULL)
        delete [] label_arr;
}


bool TrafficTestLoader::check_file_existence(const string& filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}


// Empty tensors are allowed as parameters. The shape will be assigned to the two tensors.
int TrafficTestLoader::next_batch(TypedTensor& inputs, Tensor<int>& labels)
{   // load batched inputs and labels to the provided arguments, return the number of samples
    // Since testing scans the images only once, so we can convert uchar pixels to doubleing numbers on-the-fly
    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    DTYPE *inputs_ptr = inputs.get_pointer();
    int *labels_ptr = labels.get_pointer();

    // allocate storage for inputs and labels if they have none
    if (inputs_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for inputs tensor\n");
        inputs_ptr = new DTYPE[mbatch_size * total_elems];
        inputs.set_pointer(inputs_ptr, true);
        inputs.set_shape({mbatch_size, image_shape[0], image_shape[1], image_shape[2]});
    }
    if (labels_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for labels tensor\n");
        labels_ptr = new int[mbatch_size];
        labels.set_pointer(labels_ptr, true);
        labels.set_shape({mbatch_size});
    }

    // compute no. of samples of this batch
    const int n_samples = min((batch_idx + 1) * mbatch_size, n_images) - batch_idx * mbatch_size;
    // cout << "n_samples: " << n_samples << endl;

    // return batch_idx-th batch, assume storage in `inputs` and `labels` are pre-allocated
    // CAUTION: NO SPACE CHECK! shape should be inputs: (B, C, H, W), labels: (B)
    // const vector<int> inputs_shape = inputs.get_shape();
    // const vector<int> labels_shape = labels.get_shape();
    // if (inputs_shape[0] < n_samples || inputs_shape[1] != image_shape[0] || 
    //     inputs_shape[2] != image_shape[1] || inputs_shape[3] != image_shape[2] || 
    //     labels_shape[0] < n_samples)
    // {
    //     throw invalid_argument("Incompatible shape!\n");   
    // }

    // convert uchar pixels to doubleing numbers on-the-fly, copy results to the tensor storage
    for (int i = 0; i < n_samples * total_elems; ++ i)
    {
        // Note: no scaling according to my pytorch implementation, please refer to poc/model/train.py
        inputs_ptr[i] = (DTYPE)image_arr[batch_idx * mbatch_size * total_elems + i]; // / 255.;
    }
    inputs.set_shape({n_samples, image_shape[0], image_shape[1], image_shape[2]});

    for (int i = 0; i < n_samples; ++ i)
    {
        labels_ptr[i] = label_arr[batch_idx * mbatch_size + i];
    }
    labels.set_shape({n_samples});

    batch_idx += 1;

    return n_samples;
}



CIFAR10TestLoader::CIFAR10TestLoader(string root, int batch_size, string dump_root, bool force_reload):
    n_images(10000), mbatch_size(batch_size), 
    image_arr(NULL), label_arr(NULL), image_shape({3, 32, 32}),
    batch_idx(0)
{
    const string image_dump_path = dump_root + "images.dmp";
    const string label_dump_path = dump_root + "labels.dmp";

    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    image_arr = new double[(n_images) * total_elems];
    label_arr = new int[n_images];
    double mean[3] = {0.4914, 0.4822, 0.4465};
    double std[3] = {0.2023, 0.1994, 0.2010};

    if (force_reload or not check_file_existence(image_dump_path) or not check_file_existence(label_dump_path))
    {
        ifstream anno_in;
        char image_filename[10];        // store the image name
        int width, height, roi_x1, roi_y1, roi_x2, roi_y2;
        int classid;
        Mat original_img, resized_img;
        int target_height = image_shape[1], target_width = image_shape[2];
        int label_arr_idx = 0;
        
        vector<string> imgs_paths; 
    
	    struct dirent *sub_dir = nullptr;
        DIR *dp = nullptr;
        // dp = opendir("../data/cifar10/test");
        dp = opendir(root.c_str());
        if (dp != nullptr) {
            while ((sub_dir = readdir(dp))){
		        if (((string)sub_dir->d_name) == ".." || ((string) sub_dir->d_name) == ".")
        			continue;
                //printf ("subdir: %s\n", sub_dir->d_name);
		        string subdir_path = root + ((string) sub_dir->d_name);
                //printf ("subdir path: %s\n", subdir_path.c_str());
		
                int path_idx = 0;
		        DIR *sub_dp = nullptr;
		        struct dirent *filename = nullptr;
		        sub_dp = opendir(subdir_path.c_str());
		        if(sub_dp != nullptr){
			        while((filename = readdir(sub_dp))){
				        if ((path_idx++) < 2)
					        continue;
				        string file_path = root + ((string) sub_dir->d_name) + "/" + ((string) filename->d_name);
				        //printf("filename: %s\n", file_path.c_str());
				        imgs_paths.push_back(file_path);
                        label_arr[label_arr_idx++] = ((string)(sub_dir->d_name)).c_str()[0] - '0';
			        }	
		        }
		        closedir(sub_dp);
  	        }
        }
        closedir(dp);
        cout << "number of image paths: " << imgs_paths.size() <<endl; 
        // random_shuffle(imgs_paths.begin(), imgs_paths.end());
        //for(int i = 0; i < n_images; i++){
        //    cout << imgs_paths[i] << std::endl;
        //}

        PRINT_DEBUG("Loading images ...\n");
        for (int image_cnt = 0; image_cnt < n_images; ++ image_cnt)
        {
            original_img = imread(imgs_paths[image_cnt]); // BGR image 
            
            cv::Mat channels[3];
            cv::split(original_img, channels);
            cv::Mat temp = channels[0];
            channels[0] = channels[2];
            channels[2] = temp;
            cv::Mat output;
            cv::merge(channels, 3, output);

            resize(output, resized_img, Size(target_width, target_height), INTER_LINEAR);
            // imshow("Original image", original_img);
            // imshow("Resized image", resized_img);
            // waitKey(0);
            // // destroyWindow("window"); // does not work
            // destroyAllWindows();

            // just copy the resized image into the storage
            int idx = 0;
            //for (int c = image_shape[0]-1; c >= 0; -- c)
            for (int c = 0; c < image_shape[0]; ++ c)
            {
                for (int i = 0; i < image_shape[1]; ++ i)
                {
                    for (int j = 0; j < image_shape[2]; ++ j)
                    {
        		      	double tmp = ((unsigned int)resized_img.at<Vec3b>(i, j)[c]) / 255.0;     
                        image_arr[image_cnt * total_elems + (idx ++)] = (tmp - mean[c]) / std[c];
                    }
                }
            }
	    // cout << imgs_paths[image_cnt] << endl;
	    // char path_label = imgs_paths[image_cnt][21];
	    // cout << imgs_paths[image_cnt][21] << imgs_paths[image_cnt][22] << imgs_paths[image_cnt][23] << endl;
	    // printf("label path: %c\n", path_label);
            // label_arr[image_cnt] = path_label - '0';
            //cout << label_arr[image_cnt] << endl;
            //label_arr[image_cnt] = 0;

            if (image_cnt % 1000 == 0)
            {
                if (DEBUG_FLAG) printf("[%d / %d] finished\n", image_cnt, n_images);
            }
        }

        anno_in.close();

        // dump the loaded data
        PRINT_DEBUG("Saving loaded data ... ");
        ofstream out_images(image_dump_path, ios::out | ios::binary);
        ofstream out_labels(label_dump_path, ios::out | ios::binary);

        out_images.write((const char *)image_arr, n_images * total_elems * sizeof(double) / sizeof(char));
        out_labels.write((const char *)label_arr, n_images * sizeof(int) / sizeof(char));

        out_images.close();
        out_labels.close();

        PRINT_DEBUG("Done!\n");
    }
    else
    {
        PRINT_DEBUG("Loading saved data ... ");
        ifstream in_images(image_dump_path, ios::in | ios::binary);
        ifstream in_labels(label_dump_path, ios::in | ios::binary);

        in_images.read((char *)image_arr, n_images * total_elems * sizeof(double) / sizeof(char));
        in_labels.read((char *)label_arr, n_images * sizeof(int) / sizeof(char));

        in_images.close();
        in_labels.close();

        PRINT_DEBUG("Done!\n");
    }

    // cout << (int)image_arr[10086] << "\t" << label_arr[10086] << endl; // should be: 139  7
}


CIFAR10TestLoader::~CIFAR10TestLoader()
{
    if (image_arr != NULL)
        delete [] image_arr;
    if (label_arr != NULL)
        delete [] label_arr;
}


bool CIFAR10TestLoader::check_file_existence(const string& filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}


// Empty tensors are allowed as parameters. The shape will be assigned to the two tensors.
int CIFAR10TestLoader::next_batch(TypedTensor& inputs, Tensor<int>& labels)
{   // load batched inputs and labels to the provided arguments, return the number of samples
    // Since testing scans the images only once, so we can convert uchar pixels to doubleing numbers on-the-fly
    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    DTYPE *inputs_ptr = inputs.get_pointer();
    int *labels_ptr = labels.get_pointer();

    // allocate storage for inputs and labels if they have none
    if (inputs_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for inputs tensor\n");
        inputs_ptr = new DTYPE[mbatch_size * total_elems];
        inputs.set_pointer(inputs_ptr, true);
        inputs.set_shape({mbatch_size, image_shape[0], image_shape[1], image_shape[2]});
    }
    if (labels_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for labels tensor\n");
        labels_ptr = new int[mbatch_size];
        labels.set_pointer(labels_ptr, true);
        labels.set_shape({mbatch_size});
    }

    // compute no. of samples of this batch
    const int n_samples = min((batch_idx + 1) * mbatch_size, n_images) - batch_idx * mbatch_size;
    // cout << "n_samples: " << n_samples << endl;

    // return batch_idx-th batch, assume storage in `inputs` and `labels` are pre-allocated
    // CAUTION: NO SPACE CHECK! shape should be inputs: (B, C, H, W), labels: (B)
    // const vector<int> inputs_shape = inputs.get_shape();
    // const vector<int> labels_shape = labels.get_shape();
    // if (inputs_shape[0] < n_samples || inputs_shape[1] != image_shape[0] || 
    //     inputs_shape[2] != image_shape[1] || inputs_shape[3] != image_shape[2] || 
    //     labels_shape[0] < n_samples)
    // {
    //     throw invalid_argument("Incompatible shape!\n");   
    // }

    // convert uchar pixels to doubleing numbers on-the-fly, copy results to the tensor storage
    for (int i = 0; i < n_samples * total_elems; ++ i)
    {
        // Note: no scaling according to my pytorch implementation, please refer to poc/model/train.py
        inputs_ptr[i] = (DTYPE)image_arr[batch_idx * mbatch_size * total_elems + i]; // / 255.;
    }
    inputs.set_shape({n_samples, image_shape[0], image_shape[1], image_shape[2]});

    for (int i = 0; i < n_samples; ++ i)
    {
        labels_ptr[i] = label_arr[batch_idx * mbatch_size + i];
    }
    labels.set_shape({n_samples});

    batch_idx += 1;

    return n_samples;
}


ImageNetTestLoader::ImageNetTestLoader(string root, int batch_size, string dump_root, bool force_reload):
    n_images(4000), mbatch_size(batch_size), load_gra(2000), mdump_root(dump_root), image_arr(NULL), label_arr(NULL), image_shape({3, 224, 224}), batch_idx(0), cur_set(0)
{
    PRINT_DEBUG("Read images ...\n");
    vector<instance> dataset = make_dataset(root);
    PRINT_DEBUG("Read images is ok!\n");
    
    PRINT_DEBUG("Preparing dataset ...\n");
    int set_nums = int(n_images/load_gra);
    // int indivision = n_images % load_gra; 
    int set_idx;
    for (set_idx = 0; set_idx < set_nums; set_idx++){

        save_dump_data(dump_root, set_idx, image_shape, load_gra, dataset);

    }
    PRINT_DEBUG("Dataset is ok!\n");
    // if(indivision){
    //     save_dump_data(dump_root, set_idx+1, image_shape, indivision, force_reload, dataset, image_arr_set, label_arr_set);
    // }
    // cout << (int)image_arr[10086] << "\t" << label_arr[10086] << endl; // should be: 139  7
    PRINT_DEBUG("Initialize current sub_dataset ... ");
    const string image_dump_path = mdump_root + "images_0" + ".dmp";
    const string label_dump_path = mdump_root + "labels_0" + ".dmp";

    ifstream in_images(image_dump_path, ios::in | ios::binary);
    ifstream in_labels(label_dump_path, ios::in | ios::binary);

    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    image_arr = new double[(load_gra) * total_elems];
    label_arr = new int[load_gra];
    in_images.read((char *)image_arr, load_gra * total_elems * sizeof(double) / sizeof(char));
    in_labels.read((char *)label_arr, load_gra * sizeof(int) / sizeof(char));
    in_images.close();
    in_labels.close();
    PRINT_DEBUG("Initialize current sub_dataset is ok!\n");
}


ImageNetTestLoader::~ImageNetTestLoader()
{
    // if (image_arr != NULL)
    //     delete [] image_arr;
    // if (label_arr != NULL)
    //     delete [] label_arr;
}


bool ImageNetTestLoader::check_file_existence(const string& filename)
{
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}


// Empty tensors are allowed as parameters. The shape will be assigned to the two tensors.
int ImageNetTestLoader::next_batch(TypedTensor& inputs, Tensor<int>& labels)
{   // load batched inputs and labels to the provided arguments, return the number of samples
    // Since testing scans the images only once, so we can convert uchar pixels to doubleing numbers on-the-fly
    const int total_elems = image_shape[0] * image_shape[1] * image_shape[2];
    DTYPE *inputs_ptr = inputs.get_pointer();
    int *labels_ptr = labels.get_pointer();

    // allocate storage for inputs and labels if they have none
    if (inputs_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for inputs tensor\n");
        inputs_ptr = new DTYPE[mbatch_size * total_elems];
        inputs.set_pointer(inputs_ptr, true);
        inputs.set_shape({mbatch_size, image_shape[0], image_shape[1], image_shape[2]});
    }
    if (labels_ptr == NULL)
    {
        PRINT_DEBUG("Set new pointer for labels tensor\n");
        labels_ptr = new int[mbatch_size];
        labels.set_pointer(labels_ptr, true);
        labels.set_shape({mbatch_size});
    }
    // int set_nums = int(n_images / load_gra);
    // int indivision = n_images % load_gra;
    // if (indivision)
    //     set_nums += 1;
    // cout << "1 " <<endl;
    int set_idx = int(batch_idx * mbatch_size / load_gra);
    // cout << "2 " << set_idx << " " << cur_set <<endl;
    if (set_idx == cur_set +1){
        // cout << "3 " << set_idx << " " << cur_set <<endl;
        delete [] image_arr;
        delete [] label_arr;
        PRINT_DEBUG("Switching subdataset ... ");
        const string image_dump_path = mdump_root + "images_" + std::to_string(set_idx) + ".dmp";
        const string label_dump_path = mdump_root + "labels_" + std::to_string(set_idx) + ".dmp";

        ifstream in_images(image_dump_path, ios::in | ios::binary);
        ifstream in_labels(label_dump_path, ios::in | ios::binary);
        
        image_arr = new double[(load_gra) * total_elems];
        label_arr = new int[load_gra];
        in_images.read((char *)image_arr, load_gra * total_elems * sizeof(double) / sizeof(char));
        in_labels.read((char *)label_arr, load_gra * sizeof(int) / sizeof(char));
        in_images.close();
        in_labels.close();
        printf("New sub_dataset: %d is ok!\n", set_idx);
        cur_set += 1;
        cout << "check cur_set: " << cur_set <<endl;

    }
    // cout << "4 " <<endl;
    int passed_batches = int(cur_set * load_gra / mbatch_size);
    // cout << "5 " << passed_batches <<endl;
    int idx_inset = batch_idx - passed_batches;
    // cout << "6 " << idx_inset <<endl;
    // int idx_inset = batch_idx - set_idx * (load_gra / mbatch_size);
    // printf("set_id: %d, idx_inset: %d", set_idx, idx_inset);
    // cout << "7 " << set_idx << " " << idx_inset <<endl;
    // compute no. of samples of this batch
    // const int n_samples = min((batch_idx + 1) * mbatch_size, n_images) - batch_idx * mbatch_size;
    const int n_samples = mbatch_size;
    // cout << "n_samples: " << n_samples << endl;

    // convert uchar pixels to doubleing numbers on-the-fly, copy results to the tensor storage
    for (int i = 0; i < n_samples * total_elems; ++ i)
    {
        // Note: no scaling according to my pytorch implementation, please refer to poc/model/train.py
        inputs_ptr[i] = (DTYPE)image_arr[idx_inset * mbatch_size * total_elems + i]; // / 255.;
    }
    // cout << "8 " <<endl;
    inputs.set_shape({n_samples, image_shape[0], image_shape[1], image_shape[2]});

    for (int i = 0; i < n_samples; ++ i)
    {
        labels_ptr[i] = label_arr[idx_inset * mbatch_size + i];
    }
    // cout << "9 " <<endl;
    labels.set_shape({n_samples});
    batch_idx += 1;
    return n_samples;
}





// int main(void)
// {
//     // TrafficTestLoader loader("../data/traffic/", 2, "./dump/", true);
//     int image_shape[] = {3, 32, 32};
//     int target_height = image_shape[1], target_width = image_shape[2];
//     Mat original_img, resized_img;
//     original_img = imread("../data/cifar10/test/0/0_3.png");
//     // 分离彩色图像的三个通道
//     cv::Mat channels[3];
//     cv::split(original_img, channels);

//     // 将红色通道和蓝色通道交换
//     cv::Mat temp = channels[0];
//     channels[0] = channels[2];
//     channels[2] = temp;

//     // 合并三个通道为一个彩色图像
//     cv::Mat output;
//     cv::merge(channels, 3, output);

//     resize(output, resized_img, Size(target_width, target_height), INTER_LINEAR);
    
//     printf("%d %d\n", resized_img.size().height, resized_img.size().width);
//     printf("%d %d %d\n", resized_img.at<Vec3b>(0, 0)[0], resized_img.at<Vec3b>(0, 0)[1], resized_img.at<Vec3b>(0, 0)[2]);
//     printf("%d %d %d\n", resized_img.at<Vec3b>(6, 6)[1], resized_img.at<Vec3b>(23, 27)[0], resized_img.at<Vec3b>(18, 50)[2]);

//     // int idx = 0;
//     // //for (int c = image_shape[0]-1; c >= 0; -- c)
//     // for (int c = 0; c < image_shape[0]; ++ c)
//     // {
//     //     for (int i = 0; i < image_shape[1]; ++ i)
//     //     {
//     //         for (int j = 0; j < image_shape[2]; ++ j)
//     //         {
//     //     double tmp = ((unsigned int)resized_img.at<Vec3b>(i, j)[c]) / 255.0;     
//     //             image_arr[image_cnt * total_elems + (idx ++)] = (tmp - mean[c]) / std[c];
//     //         }
        
//     //     }
//     // }



//     // CIFAR10TestLoader test_loader("../data/cifar10/", 2, "../infras/dump/cifar10/", true);

//     // TypedTensor inputs;
//     // Tensor<int> labels;

//     // loader.next_batch(inputs, labels);
//     // cout << "Inputs\n" << inputs << endl;
//     // cout << "Labels\n" << labels << endl;

//     return 0;
// }