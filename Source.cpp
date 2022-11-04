///////////EXAMNE GIT
#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace cv;
using namespace std;


////////////////////    MOSTRAR IMAGENES   //////////////////// 
Mat cargarImg(char NombreImg[]) {

    Mat imagen; // Matriz que contiene nuestra imagen 


    /*********Lectura de la imagen*********/
    imagen = imread(NombreImg);

    if (!imagen.data)
    {
        cout << "Error al cargar la imagen: " << NombreImg << endl;
        exit(1);
    }
    return imagen;
}

void mostrarImg(string windowName, Mat img) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, img);
}
////////////////////////////////////////////////////////////////////////////


////////////////////    CREAR KERNEL   //////////////////// 
vector<vector<float>> generarKernel(float sigma, int kernel) {

    int centro = (kernel - 1) / 2;          //ubica el centro
    float r, s, z, gaussResultado;
    float sum = 0.0;

    vector<vector<float>> mascara(kernel, vector<float>(kernel, 0));

    s = 2.0 * sigma * sigma;
    cout << "\n"<< endl;
    cout << "KERNEL" << endl;
    //recorremos para llenar la mascara de los valores correspondientes
    for (int x = -centro; x <= centro; x++) {
        for (int y = -centro; y <= centro; y++) {
            //G(x,y)=[1/2pi(sigma)^2][e^-{(x^2+y^2)/2sigma}]
            r = sqrt(x * x + y * y);
            z = (exp(-(r * r) / s)) / (M_PI * s);
            gaussResultado = (exp(-(r * r) / s)) / (M_PI * s);
            mascara[x + centro][y + centro] = gaussResultado;
            cout << gaussResultado << "\t";
            sum += gaussResultado;

        }
        cout << "\n";
    }

    //mascara generada
    for (int i = 0; i < kernel; i++) {
        for (int j = 0; j < kernel; j++) {
            mascara[i][j] /= sum;
            //cout << mascara[i][j] << endl;
        }
    }

    return mascara;
}

Mat RellenoKernel(int filas, int columnas, int kernel)
{
    int diferenciaBordes = kernel - 1;
    Mat matriz(filas + diferenciaBordes, columnas + diferenciaBordes, CV_8UC1);

    //recorremos la imgen a partir de sus filas y columnas para añadir bordes
    for (int i = 0; i < filas + diferenciaBordes; i++)
    {
        for (int j = 0; j < columnas + diferenciaBordes; j++)
        {
            matriz.at<uchar>(Point(j, i)) = uchar(0);
        }
    }

    return matriz;
}


Mat ImgARelleno(Mat bordes, Mat original, int kernel)
{
    int diferenciaBordes = ((kernel - 1) / 2);
    int filas = bordes.rows;
    int columnas = bordes.cols;

    for (int i = diferenciaBordes; i < filas - diferenciaBordes; i++)
    {
        for (int j = diferenciaBordes; j < columnas - diferenciaBordes; j++)
        {
            bordes.at<uchar>(Point(j, i)) = original.at<uchar>(Point(j - diferenciaBordes, i - diferenciaBordes));
        }
    }

    return bordes;
}
//////////////////////////////////////////////////////////////

////////////////////    APLICAR FILTRO   //////////////////// 
float convolucion(Mat imgBordes, vector<vector<float>> mascara, int size, int x, int y) {

    int limites = (size - 1) / 2;

    int filasImgBordes = imgBordes.rows;
    int columnasImgBordes = imgBordes.cols;

    float sumatoriaFiltro = 0.0;

    for (int i = -limites; i <= limites; i++) {
        for (int j = -limites; j <= limites; j++) {

            float valMascara = mascara[i + limites][j + limites];
            int coordY = y + j + limites;
            int coordX = x + i + limites;

            float valImagen = imgBordes.at<uchar>(coordY, coordX);

            sumatoriaFiltro += valMascara * valImagen;
        }
    }

    return sumatoriaFiltro;
}

Mat aplicarFiltro(Mat imagenOriginal, Mat matrizConBordes, vector<vector<float>> mascara, int kernel) {
    int filas = imagenOriginal.rows;
    int columnas = imagenOriginal.cols;

    Mat imagenFiltroAplicado(filas, columnas, CV_8UC1);

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            float val = abs(static_cast<int>(convolucion(matrizConBordes, mascara, kernel, i, j)));
            imagenFiltroAplicado.at<uchar>(Point(i, j)) = val;
        }
    }

    return imagenFiltroAplicado;
}

//////////////////////////////////////////////////////////////


////////////////////    FILTRO SOBEL   //////////////////// 
//////     GRADENTES
//tomamos como ejemplo la generacion del Kernel con vector<vector<
//a cada uno se asigna el valr correspodiente de la gradientes
vector<vector<float>> grad_Gx() {
    //vector que tiene los valores de la Gx
    vector<vector<float>> gradiente(3, vector<float>(3, 0));
    //valores Gx
    gradiente[0][0] = -1;
    gradiente[0][1] = 0;
    gradiente[0][2] = 1;

    gradiente[1][0] = -2;
    gradiente[1][1] = 0;
    gradiente[1][2] = 2;

    gradiente[2][0] = -1;
    gradiente[2][1] = 0;
    gradiente[2][2] = 1;

    return gradiente;
}

vector<vector<float>> grad_Gy() {
    //vector que tiene los valores de la Gy
    vector<vector<float>> gradiente(3, vector<float>(3, 0));
    //valores Gy
    gradiente[0][0] = -1;
    gradiente[0][1] = -2;
    gradiente[0][2] = -1;

    gradiente[1][0] = 0;
    gradiente[1][1] = 0;
    gradiente[1][2] = 0;

    gradiente[2][0] = 1;
    gradiente[2][1] = 2;
    gradiente[2][2] = 1;

    return gradiente;
}


Mat Sobel(Mat imgGx, Mat imgGy) {

    int filas = imgGx.rows;
    int columnas = imgGy.cols;

    Mat sobel(filas, columnas, CV_8UC1);

    float u = 127.5;            //valor maximo para ubralado
    double  d;       //varibes para distancia
    

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {

            double Gx_value = imgGx.at<uchar>(Point(j, i));
            double Gy_value = imgGy.at<uchar>(Point(j, i));

            double intensidad = sqrt(pow(Gx_value, 2) + pow(Gy_value, 2));
            intensidad = static_cast<int>(intensidad);
            if (intensidad > u) {
                intensidad = 255;
            }
            else {
                intensidad = 0;
            }

            sobel.at<uchar>(Point(j, i)) = uchar(intensidad);
            d = atan(Gy_value / Gx_value);
        }
    }

    return sobel;
}



////////////////////    ECUALIZAR   //////////////////// 

Mat ecualizar(Mat imagen) {
    int rows = imagen.rows; 
    int cols = imagen.cols;
    Mat ecualizada(rows, cols, CV_8UC1);    //matriz que guarda imagen ecualizada


    double* mejor = new double[rows * cols];
    //sumamos la 1 con la anterior

    double sum=0.0; 
    int c = 0;
    //recorremos nuestra imagen para calcular el histograma
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            sum += imagen.at<uchar>(Point(j, i));  //acumulamos sumatorias      //
            mejor[c] = imagen.at<uchar>(Point(j, i));
            c++;
        }
    }

    
    double max = 0, min = 10000;    //obtencion de max y min
    for (int i = 0; i < rows * cols; i++) {
        if (mejor[i] < min && mejor[i] != 0) {  //comparamos imagen mejorada contra los valores max y min
            min = mejor[i]; //asignamos nuevo valor
        }
        if (mejor[i] > max) {
            max = mejor[i];
        }
    }

    c = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ecualizada.at<uchar>(Point(j, i)) = (mejor[c] - min) * 255 / (max - min); //formula para ecualizar  
            c++;
        }
    }


    return ecualizada;
}


////////////////////    CANNY   ////////////////////
vector<vector<float>> grad_Hx() {
    //vector que tiene los valores de la Gx
    vector<vector<float>> mascara(3, vector<float>(3, 0));
    //valores Gx
    mascara[0][0] = -1;
    mascara[0][1] = 0;
    mascara[0][2] = 1;

    mascara[1][0] = -2;
    mascara[1][1] = 0;
    mascara[1][2] = 2;

    mascara[2][0] = -1;
    mascara[2][1] = 0;
    mascara[2][2] = 1;

    return mascara;
}

vector<vector<float>> grad_Hy() {
    //vector que tiene los valores de la Gy
    vector<vector<float>> mascara(3, vector<float>(3, 0));
    //valores Gy
    mascara[0][0] = 1;
    mascara[0][1] = 2;
    mascara[0][2] = 1;

    mascara[1][0] = 0;
    mascara[1][1] = 0;
    mascara[1][2] = 0;

    mascara[2][0] = -1;
    mascara[2][1] = -2;
    mascara[2][2] = -1;

    return mascara;
}


Mat FCanny(Mat imgGx, Mat imgGy) {

    int filas = imgGx.rows;
    int columnas = imgGy.cols;

    Mat canny(filas, columnas, CV_8UC1);

    float umbral = 127.5;
    double intensidad, direccion;
    double valGx, valGy;

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {

            valGx = imgGx.at<uchar>(Point(j, i));
            valGy = imgGy.at<uchar>(Point(j, i));

            intensidad = sqrt(pow(valGx, 2) + pow(valGy, 2));
            intensidad = static_cast<int>(intensidad);
            if (intensidad > umbral) {
                intensidad = 255;
            }
            else {
                intensidad = 0;
            }

            canny.at<uchar>(Point(j, i)) = uchar(intensidad);
            //direccion = atan(valor_y / valor_x);
        }
    }

    return canny;
}


////////////////////    OBTENER IMPRIMIR TAMAÑO IMG   //////////////////// 

void size_img(string name,Mat img) {
    cout << "\t"<< name << "\t";
    cout << img.rows << " X ";
    cout << img.cols << endl;
}



int main()
{
    char NombreImg[] = "lena.png";
    Mat imagen = cargarImg(NombreImg);      //lectura de imagen
    Mat imagen_gray;

    float sigma;              //variable sigma
    int kernel;               //variable kernel (tamaño)


    cout << "Digite el tamano del Kernel: ";
    cin >> kernel;

    //Validacion de que el kernel sea de tamaño impar
    while (kernel < 1 || kernel % 2 == 0) {
        cout << "Error! el kernel debe ser de tamano impar ";

        cin >> kernel;
    }

    cout << "Digite el valor de 'sigma': ";
    cin >> sigma;



    cvtColor(imagen, imagen_gray, COLOR_BGR2GRAY); //convertimos a escala de grises
    int filasImagen = imagen_gray.rows;     //obtemos el tamaño de la imagen original
    int columnasImagen = imagen_gray.cols;

    vector<vector<float>> m_kernel = generarKernel(sigma, kernel);  //mascara generada

    ////////////////////    RELLENO DE BORDES   //////////////////// 

    Mat imgBordes = RellenoKernel(filasImagen, columnasImagen, kernel);     //matriz con bordes añadidos
    imgBordes = ImgARelleno(imgBordes, imagen_gray, kernel);

    ////////////////////    FILTRO GAUSSIANO   //////////////////// 
    Mat imagenGaussiano = aplicarFiltro(imagen, imgBordes, m_kernel, kernel); //aplica filtro gaussiano

    imgBordes = RellenoKernel(filasImagen, columnasImagen, kernel);
    imgBordes = ImgARelleno(imgBordes, imagenGaussiano, kernel);

    ////////////////////    SOBEL   //////////////////// 
    vector<vector<float>> Gx = grad_Gx();       //varaibles que guardan las gradientes
    vector<vector<float>> Gy = grad_Gy();

    Mat imgGx = aplicarFiltro(imagenGaussiano, imgBordes, Gx, 3);   //aplicamos gradiantes GX y GY a la imagen
    Mat imgGy = aplicarFiltro(imagenGaussiano, imgBordes, Gy, 3);

    Mat imgsobel = Sobel(imgGy, imgGx);  //suman ambas gradientes
   


    ////////////////////    IMAGEN ECUALIZADA   //////////////////// 
    Mat imgEcualizada = ecualizar(imagen_gray);

    ////////////////////    CANNY   //////////////////// 
    vector<vector<float>> Hx = grad_Hx();       //varaibles que guardan las gradientes
    vector<vector<float>> Hy = grad_Hy();

    Mat imgHx = aplicarFiltro(imagenGaussiano, imgBordes, Hx, 3);   //aplicamos gradiantes GX y GY a la imagen
    Mat imgHy = aplicarFiltro(imagenGaussiano, imgBordes, Hy, 3);

    Mat imgcanny = FCanny(imgHy, imgHx);  //suman ambas gradientes

    Mat imgCanny;
    //Canny(imgcanny, imgCanny, 100, 200, Canny, false);
    Canny(imagenGaussiano, imgCanny, 100, 200, 3, false);
    


    ////////////////////    TAMAÑO DE C/IMAGENES   //////////////////// 
    cout << "\nTAMANO DE IMAGENES" << endl;
    size_img("Imagen Original\t", imagen);
    size_img("Imagen con Bordes", imgBordes);
    size_img("Imagen Filtro Gaussiano", imagenGaussiano);
    //size_img("Imagen con Filtro Sobel", imgsobel);
    size_img("Imagen Ecualizada", imgEcualizada);
    

    ////////////////////    MOSTRAR IMAGENES   //////////////////// 
    mostrarImg("Imagen Original", imagen);
    mostrarImg("Imagen Escala de Grises", imagen_gray);
    mostrarImg("Imagen con Bordes", imgBordes);
    mostrarImg("Imagen con Filtro Gaussiano", imagenGaussiano);
    mostrarImg("Imagen con Filtro Sobel |Gx|", imgsobel);
    mostrarImg("Imagen Ecualizada", imgEcualizada);
    mostrarImg("Imagen con Filtro Canny", imgCanny);
   
   

    waitKey(0);

    return 1;
}