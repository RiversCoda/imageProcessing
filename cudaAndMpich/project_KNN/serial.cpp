#include<iostream>  
#include<map>  
#include<vector>  
#include<stdio.h>  
#include<cmath>  
#include<algorithm>  
#include<fstream>  
#include<cstdlib>  
#include<stdlib.h>  
using namespace std;  
  
#define maxRow 1700//训练数据行数  
#define maxCol 20//列数（数据的维度）  
#define testRow 30//测试数据个数  
int TDnum = 0;//测试数据的个数统计  
double testData_glo[testRow][maxCol];//存放测试数据集  
string label[testRow];//存放测试数据集标签  
string label1[testRow];//存放测试数据集预测得标签  
  
float gpu_time = 0;  
  
  
typedef pair<int, double> PAIR;//模板类  
ifstream fin;//输入文件流  
  
class KNN//KNN类  
{  
private:  
    double dataSet[maxRow][maxCol];//训练数据数组  
    string   labels[maxRow];//训练数据的标签数组  
    double testData[maxCol];//一个测试数据数组  
    map<int, double> map_index_dis;//map类模板存储训练数据的索引值和到测试数据的距离，key是数据的编号/序号，value是距离  
    map<string, int> map_label;//map类模板存储训练数据的标签和索引  
public:  
    int k = 5;//初始化KNN算法中的k值大小，k值一般取奇数且一般小于20  
    KNN(char* filename);//有参构造函数  
    void get_all_distance();//获得欧式距离函数  
    double get_distance(double* d1, double* d2);  
  
    string get_max_fre_label();//找到前k个距离最接近标签函数  
    void get_error_rate();//误差率  
    struct CmpByValue//map的pair对按距离大小排序的结构体  
    {  
        bool operator() (const PAIR& lhs, const PAIR& rhs)  
        {  
            return lhs.second < rhs.second;  
        }  
  
    };  
  
};  
KNN::KNN(char* filename)  
{  
    if (TDnum == 0) {  
        fin.open(filename);//打开文件  
        if (!fin)//打开失败  
        {  
            cout << "can not open the file" << endl;  
            exit(0);  
        }  
        for (int i = 0; i < maxRow; i++)//读入训练数据及其标签存储在dataSet和labels中  
        {  
            for (int j = 0; j < maxCol; j++)  
            {  
                fin >> dataSet[i][j];  
  
            }  
	    fin >> labels[i];
        }  
        fin.close();//关闭读操作  
    }  
}  
double KNN::get_distance(double* d1, double* d2)  
{  
    double sum = 0;  
    for (int i = 0; i < maxCol; i++)  
    {  
        sum = sum + (d1[i] - d2[i]) * (d1[i] - d2[i]); 
    }  
    return sum;  
}  
void KNN::get_all_distance()  
{  
    double distance;  
    int i;
    for (int j = 0; j < maxCol; j++)//循环读取测试数据
    {
	testData[j] = testData_glo[TDnum][j];
    }  
    for (i = 0; i < maxRow; i++)  
    {  
        distance = get_distance(dataSet[i], testData);  
        map_index_dis[i] = sqrt(distance);  
    }  
  
    map<int, double>::const_iterator it = map_index_dis.begin();  
    while (it != map_index_dis.end())  
    {  
        it++;  
    }  
}  
  
  
string KNN::get_max_fre_label()  
{  
    vector<pair<int, double> >vec_index_dis(map_index_dis.begin(), map_index_dis.end());  
    sort(vec_index_dis.begin(), vec_index_dis.end(), CmpByValue());//按距离排序  
  
  
    for (int i = 0; i < k; i++)//打印距离最小k个的点的信息  
    {  
#ifdef TEST  
        cout << "index = " << vec_index_dis[i].first << " the distance= " << vec_index_dis[i].second  
            << " the label = " << labels[vec_index_dis[i].first] << endl;  
#endif // TEST  
        map_label[labels[vec_index_dis[i].first]]++;//标签是key，标签数是value  
    }  
  
  
    map<string, int>::iterator itr = map_label.begin();  
    int max_freq = 0;  
    string label;  
    while (itr != map_label.end())//寻找出现次数最多的标签  
    {  
        if (itr->second > max_freq)  
        {  
            max_freq = itr->second;  
            label = itr->first;  
        }  
        itr++;  
    }  
    for (int i = 0; i < k; i++)//恢复初始值0  
    {  
        int xuhao = vec_index_dis[i].first;//距离第i小的点的序号  
        map_label[labels[xuhao]]--;  
    }  
    label1[TDnum] = label;  
    return label;  
}  
void KNN::get_error_rate()  
{  
    int i, count = 0;  
    for (i = 0; i < testRow; i++)  
    {  
        if (label[i] != label1[i])  
        {  
            count++;  
        }  
    }  
    cout << "the error rate is = " << (double)count / (double)testRow << endl;  
    cout << "the correct rate is = " << 1.0 - (double)count / (double)testRow << endl;  
}  
  
int main()  
{  
    char* filename = "ring.txt";//训练数据文件名  
    KNN knn(filename);//创建KNN类对象  
    if (TDnum == 0) {//程序开始时导入测试数据集存放至testData_glo数组  
        filename = "ring_test.txt";  
        fin.open(filename);  
        if (!fin)  
        {  
            cout << "can not open the file" << endl;  
            exit(0);  
        }  
        for (int i = 0; i < testRow; i++)  
        {  
            for (int j = 0; j < maxCol; j++)  
            {  
                fin >> testData_glo[i][j];  
  
            }  
            fin >> label[i];  
  
        }  
    }  
  
    clock_t begin = clock();//开始时间  
    for (int e = 0; e < testRow; e++)//循环完成测试数据的标签预测  
    {  
        knn.get_all_distance();//计算与训练数据的所有距离  
        knn.get_max_fre_label();  
        TDnum++;//记录测试数据的个数  
    }  
    clock_t end = clock();//结束时间  
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;  
  
    cout << "k= " << knn.k << " result:" << endl;  
    knn.get_error_rate();  
    cout << "real    label:";  
    for (int i = 0; i < testRow; i++)  
    {  
        cout << label[i] << " ";  
    }  
    cout << endl; cout << "predict label:";  
    for (int i = 0; i < testRow; i++)  
    {  
        cout << label1[i] << " ";  
    }  
    cout << endl;  
  
    printf("CPU:Time used:%lf s\n", time_spent);  
  
    return 0;  
}  


