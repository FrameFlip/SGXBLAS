#include <iostream>
#include <cstdio>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#include <cmath>
#include "common.h"
#include "myresnet.h"
#include "mydataloader.h"

// #define PREDICTION_PATH "./out.txt"
#ifndef STOP_AT
#define STOP_AT -1
#endif
#ifndef BATCH_SIZE
#define BATCH_SIZE 256
#endif

#ifndef POC_DIR
#define POC_DIR ".."
#endif

using namespace std;


struct stat st = {0};


void eval_gtsrb(string network)
{

	// Hyperparams
	// const double alpha = 0.15;				// model scaling factor
	const int batch_size = 256;		// batch size
	const int n_classes = 43;				// total number of classification categories
	const int stop_at = STOP_AT;			// for debug, stop after this batch

	
	if (network == "resnet34"){
		printf("Initailizing ResNet34 task model\n");
		ResNet34_v2 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/gtsrb_resnet34/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// Load testing dataset
		const string testset_path = POC_DIR "/data/gtsrb/test";
		const string dump_path = POC_DIR "data/dump/gtsrb/";
		if (stat(dump_path.c_str(), &st) == -1) {
			int isCreate = ::mkdir(dump_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
			if(isCreate) cout<< "fail to create dump file!" <<endl;
		}
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		TrafficTestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		// int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);
			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}
				pred_i = task_pred_i; // always adopt the task model prediction

				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);

	}else if (network == "resnet50"){
		printf("Initailizing ResNet50 task model\n");
		ResNet50 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/gtsrb_resnet50/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// Load testing dataset
		const string testset_path = POC_DIR "/data/gtsrb/test";
		const string dump_path = POC_DIR "/data/dump/gtsrb/";
		if (stat(dump_path.c_str(), &st) == -1) {
			int isCreate = ::mkdir(dump_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
			if(isCreate) cout<< "fail to create dump file!" <<endl;
		}
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		TrafficTestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		// int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);
			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}
				pred_i = task_pred_i; // always adopt the task model prediction

				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);
	}else if(network == "vgg16"){
		printf("Initailizing VGG16 task model\n");
		VGG16 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/gtsrb_vgg16/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// Load testing dataset
		const string testset_path = POC_DIR "/data/gtsrb/test";
		const string dump_path = POC_DIR "/data/dump/gtsrb/";
		printf("check dump path: %s\n", dump_path.c_str());
		if (stat(dump_path.c_str(), &st) == -1) {
			int isCreate = ::mkdir(dump_path.c_str(), S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
			if(isCreate) cout<< "fail to create dump file!" <<endl;
		}
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		TrafficTestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		// int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);
			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}
				pred_i = task_pred_i; // always adopt the task model prediction

				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);
	}else{
		std::cout << "Not Implemented Error! Network: " << network << std::endl;
	}
}

void eval_cifar10(string network) {
	// Hyperparams
	// const double alpha = 0.15;				// model scaling factor
	const int batch_size = 50;		// batch size
	const int n_classes = 10;				// total number of classification categories
	const int stop_at = STOP_AT;			// for debug, stop after this batch

	if (network == "resnet34"){
		printf("Initailizing ResNet34 task model\n");
		ResNet34_v2 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/cifar10_resnet34/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// return;

		// Load testing dataset
		const string testset_path = POC_DIR "/data/cifar10/test/";
		const string dump_path = POC_DIR "/data/dump/cifar10/";
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		CIFAR10TestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);

			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}

				pred_i = task_pred_i; // always adopt the task model prediction


				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		// printf(
		//     "[Final] Prediction accuracy: %.2f%% ( %d / %d )\tRunning consistency: %.2f%% ( %d / %d)\n", 
		//     (double)total_corrects * 100. / n_samples, total_corrects, n_samples,
		// 	(double)total_consistents * 100. / running_samples, total_consistents, running_samples);
		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);

	}else if (network == "resnet50"){
		printf("Initailizing ResNet50 task model\n");
		ResNet50 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/cifar10_resnet50/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// return;

		// Load testing dataset
		const string testset_path = POC_DIR "/data/cifar10/test/";
		const string dump_path = POC_DIR "/data/dump/cifar10/";
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		CIFAR10TestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);

			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}

				pred_i = task_pred_i; // always adopt the task model prediction


				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		// printf(
		//     "[Final] Prediction accuracy: %.2f%% ( %d / %d )\tRunning consistency: %.2f%% ( %d / %d)\n", 
		//     (double)total_corrects * 100. / n_samples, total_corrects, n_samples,
		// 	(double)total_consistents * 100. / running_samples, total_consistents, running_samples);
		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);

	}else if(network == "vgg16"){
		printf("Initailizing VGG16 task model\n");
		VGG16 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/cifar10_vgg16/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// return;

		// Load testing dataset
		const string testset_path = POC_DIR "/data/cifar10/test/";
		const string dump_path = POC_DIR "/data/dump/cifar10/";
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		CIFAR10TestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);

			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}

				pred_i = task_pred_i; // always adopt the task model prediction


				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		// printf(
		//     "[Final] Prediction accuracy: %.2f%% ( %d / %d )\tRunning consistency: %.2f%% ( %d / %d)\n", 
		//     (double)total_corrects * 100. / n_samples, total_corrects, n_samples,
		// 	(double)total_consistents * 100. / running_samples, total_consistents, running_samples);
		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);

	}else{
		std::cout << "Not Implemented Error! Network: " << network << std::endl;
	}
	
}

void eval_imagenet(string network) {
	// Hyperparams
	// const double alpha = 0.15;				// model scaling factor
	const int batch_size = 200;		// batch size
	const int n_classes = 1000;				// total number of classification categories
	const int stop_at = STOP_AT;			// for debug, stop after this batch

	if (network == "resnet34"){
		printf("Initailizing ResNet34 task model\n");
		ResNet34_v2 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/imagenet_resnet34/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// return;

		// Load testing dataset
		const string testset_path = POC_DIR "/data/imagenet/val/";
		const string dump_path = POC_DIR "/data/dump/imagenet/";
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		ImageNetTestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);

			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// print_vector(outputs.get_shape());

			// cout << "Inputs:\n" << inputs << endl;
			// cout << "Labels:\n" << labels << endl;
			// cout << "Outputs:\n" << outputs << endl;

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}

				pred_i = task_pred_i; // always adopt the task model prediction


				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);

	}else if (network == "resnet50"){
		printf("Initailizing ResNet50 task model\n");
		ResNet50 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/imagenet_resnet50/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// return;

		// Load testing dataset
		const string testset_path = POC_DIR "/data/imagenet/val/";
		const string dump_path = POC_DIR "/data/dump/imagenet/";
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		ImageNetTestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);

			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// print_vector(outputs.get_shape());

			// cout << "Inputs:\n" << inputs << endl;
			// cout << "Labels:\n" << labels << endl;
			// cout << "Outputs:\n" << outputs << endl;

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}

				pred_i = task_pred_i; // always adopt the task model prediction


				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);

	}else if(network == "vgg16"){
		printf("Initailizing VGG16 task model\n");
		VGG16 task_model(n_classes);
		const string task_params_path = POC_DIR "/trained_models/imagenet_vgg16/best.task.txt";
		cout << "Loading task model parameters from " << task_params_path << endl;
		ifstream task_in(task_params_path, ios::in);
		if (!task_in)
		{
			printf("Failed to open file %s\n", task_params_path.c_str());
			exit(1);
		}
		task_model.load_from(task_in);
		task_in.close();

		// return;

		// Load testing dataset
		const string testset_path = POC_DIR "/data/imagenet/val/";
		const string dump_path = POC_DIR "/data/dump/imagenet/";
		cout << "Loading testing dataset from " << testset_path << endl;
		cout << "dump path: " << dump_path << endl;
		ImageNetTestLoader test_loader(testset_path, batch_size, dump_path, false);

		// Run evaluation
		const int n_samples = test_loader.get_total_samples();
		const int n_batches = (int)ceil( (double)n_samples / batch_size );
		int total_corrects = 0;
		int total_consistents = 0;
		int running_samples = 0;

		if (stop_at >= 0)
		{
			printf(
				"[INFO] Evaluation will stop after batch %d (change `stop_at` to negative to cancel)\n", 
				stop_at + 1);
		}

		#ifdef PREDICTION_PATH
		ofstream prediction_out(PREDICTION_PATH, ios::out);
		prediction_out << "Task\tPred\tLabel" << endl;
		#endif

		TypedTensor inputs;
		Tensor<int> labels;
		for (int batch_idx = 0; batch_idx < n_batches; ++ batch_idx)
		{
			if (DEBUG_FLAG) printf("Batch [ %d / %d ]\n", batch_idx + 1, n_batches);
			int batch_samples = test_loader.next_batch(inputs, labels);

			PRINT_DEBUG("Forwarding in task model ...\n");
			TypedTensor task_outputs = task_model.forward(inputs);

			// print_vector(outputs.get_shape());

			// cout << "Inputs:\n" << inputs << endl;
			// cout << "Labels:\n" << labels << endl;
			// cout << "Outputs:\n" << outputs << endl;

			// compute correct predictions from outputs (confidence scores)
			int task_pred_i, checker_pred_i;	// predicted labels for sample i by task / checker model
			int pred_i; 						// final predicted label for sample i
			for (int i = 0; i < batch_samples; ++ i)
			{
				task_pred_i = 0;
				for (int j = 0; j < n_classes; ++ j)
				{
					if (task_outputs.at(i, j) > task_outputs.at(i, task_pred_i))
					{
						task_pred_i = j;
					}
				}

				pred_i = task_pred_i; // always adopt the task model prediction


				if (pred_i == labels.at(i))
				{
					++ total_corrects;
				}

				// write prediction to file
				#ifdef PREDICTION_PATH
				prediction_out << task_pred_i << "\t" << pred_i << "\t" << labels.at(i) << endl;
				#endif
			}

			// report stats after each batch
			running_samples += batch_samples;
			printf(
				"Batch [ %d / %d ] Running accuracy: %.2f%% ( %d / %d )\n", 
				batch_idx + 1, n_batches,
				(double)total_corrects * 100. / running_samples, total_corrects, running_samples);
			fflush(stdout);

			if (batch_idx == stop_at) break;
		}

		#ifdef PREDICTION_PATH
		prediction_out.close();
		#endif

		printf(
			"*%.4f%%\n", 
			(double)total_corrects * 100. / running_samples);

	}else{
		std::cout << "Not Implemented Error! Network: " << network << std::endl;
	}
	
}



int main(int argc, char* argv[])
{
	string dataset = argv[1];
	string network = argv[2];
	if (dataset == "gtsrb"){
		eval_gtsrb(network);
	}else if(dataset == "cifar10"){
		eval_cifar10(network);
	}else if(dataset == "imagenet"){
		eval_imagenet(network);
	}else{
		std::cout << "Not Implemented Error! Dataset: " << dataset << std::endl;
	}


	return 0;
}
