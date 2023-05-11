#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <time.h>
#include <algorithm>
#include <fstream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const int class_num = 10;
const int input_height = 32;
const int input_width = 32;
const int input_channel = 3;

const int batch_size = 1;

class Classifier {
public:
	Classifier(const wchar_t* onnx_path) {
		auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_.data(), input_.size(), input_shape_.data(), input_shape_.size());
		output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, output_.data(), output_.size(), output_shape_.data(), output_shape_.size());

		OrtSessionOptionsAppendExecutionProvider_CUDA(session_option, 0);
		session =  Ort::Session(env, onnx_path, session_option);
	}

	int set_input(string& img_paht) {
		Mat img = imread(img_paht);

		//Mat dst(input_height, input_width, CV_8UC3);
		//resize(img, dst, Size(row, col));
		//cvtColor(img, dst, COLOR_BGR2RGB);
		float* input_prt = input_.data();
		for (int c = 0; c < 3; c++) {
			for (int i = 0; i < input_height; i++) {
				for (int j = 0; j < input_width; j++) {
					float tmp = img.ptr<uchar>(i)[j * 3 + c];
					input_prt[c * input_height * input_width + i * input_width + j] = ((tmp) / 255.0 - mean_[c]) / std_[c];
				}
			}
		}
		return 0;
	}

	int forward() {
		session.Run(Ort::RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), &output_tensor_, 1);
		return 0;
	}

	int get_result(int& result) {
		result = std::distance(output_.begin(), std::max_element(output_.begin(), output_.end()));
		return 0;
	}

private:
	vector<const char*> input_names{ "img" };
	vector<const char*> output_names{ "output" };
	std::array<float, batch_size* input_height* input_width* input_channel> input_;
	std::array<float, batch_size* class_num> output_;
	std::array<int64_t, 4> input_shape_{ batch_size, input_channel, input_width, input_height };
	std::array<int64_t, 2> output_shape_{ batch_size, class_num };

	Ort::Value input_tensor_{ nullptr };
	Ort::Value output_tensor_{ nullptr };


	Ort::SessionOptions session_option;
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "赢麻了.log" };

	Ort::Session session{ nullptr };

	std::vector<float> mean_{ 0.4914, 0.4822, 0.4465 };
	std::vector<float> std_{ 0.2023, 0.1994, 0.2010 };
};

int load_img_path(string& file_path, vector<string>& img_lst, vector<int>& label_lst) {
	ifstream f(file_path.c_str());
	if (!f.is_open()) {
		cout << "文件打开失败" << endl;
		return -1;
	}
	string img_path;
	int label;
	while (getline(f, img_path)) {
		if (img_path.size() > 0) {
			img_lst.push_back(img_path);
			auto iter = img_path.find(".");
			label = std::atoi(img_path.substr(--iter, iter).c_str());
			label_lst.push_back(label);
		}
	}
	f.close();
	return 0;
}

int save_result(string& file_path, vector<int>& results) {
	ofstream f(file_path.c_str());
	if (!f.is_open()) {
		cout << "文件打开失败" << endl;
		return -1;
	}
	for (auto& res : results) {
		f << res << endl;
	}
	f.close();
	return 0;
}

float cal_acc(vector<int>& labels, vector<int>& results) {
	float TP = 0.;
	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] == results[i]) {
			TP++;
		}
	}
	return TP / labels.size();
}

int main()
{
	const wchar_t* onnx_path = L"D:/Files/projects/vs/onnxruntimelib/onnxruntime-win-x64-gpu-1.11.1/output/resnet_best.onnx";
	string img_path_file = "D:/Files/projects/Py/CNN-Backbone/data/testimg.lst";
	vector<string> img_lst;
	vector<int>  label_lst;
	vector<int> results;
	load_img_path(img_path_file, img_lst, label_lst);
	clock_t start;
	float time_cost;
	int result;
	Classifier classifier(onnx_path);

	start = clock();
	for (int i = 0; i < img_lst.size(); i++) {
		result = -1;
		classifier.set_input(img_lst[i]);
		classifier.forward();
		classifier.get_result(result);
		results.push_back(result);
	}

	time_cost = clock()-start;
	float acc = cal_acc(label_lst, results);
	std::cout << "Total Time cost: " << time_cost << "ms" << std::endl;
	std::cout << "Average Time cost: " << time_cost/img_lst.size() << "ms" << std::endl;
	std::cout << "Test Acc:  " << acc << std::endl;

	system("pause");
	return 0;
}
