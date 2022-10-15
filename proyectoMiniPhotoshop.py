from re import L
import tkinter
from tkinter import *
from tkinter import filedialog
from unittest import result
import cv2
from PIL import Image
from PIL import ImageTk
import imutils
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from tkinter import messagebox
import math

view = tkinter.Tk()
view.geometry("1000x700")
view.title("Mini Photoshop")

global imgPrincipal, imgResultado, dirImgPrincipal
global imgSecundaria, dirImgSecundaria

dirImgPrincipal = ''


def abrir_archivo():
    global imgPrincipal, dirImgPrincipal

    archivoAbierto = filedialog.askopenfilename(initialdir="/", title="Seleccione archivo", filetypes=(
        ("image", "* .jpg"), ("image files", "* .png"), ("image", "*.jpeg*")))

    dirImgPrincipal = archivoAbierto.replace('/', '\\')
    imgPrincipal = cv2.imread(archivoAbierto)
    imgPrincipal = cv2.cvtColor(imgPrincipal, cv2.COLOR_BGR2RGB)

    imgPrincipal = cv2.resize(imgPrincipal, dsize=(
        250, 250), interpolation=cv2.INTER_CUBIC)

    imgPrincipal = Image.fromarray(imgPrincipal)
    imgPrincipal = ImageTk.PhotoImage(image=imgPrincipal)

    lblImagenDeseada.configure(image=imgPrincipal)


def abrir_archivo2():
    global imgSecundaria, dirImgSecundaria

    archivoAbierto = filedialog.askopenfilename(initialdir="/", title="Seleccione archivo", filetypes=(
        ("image", "* .jpg"), ("image files", "* .png"), ("image", "*.jpeg*")))

    dirImgSecundaria = archivoAbierto.replace('/', '\\')
    imgSecundaria = cv2.imread(archivoAbierto)
    imgSecundaria = cv2.cvtColor(imgSecundaria, cv2.COLOR_BGR2RGB)

    imgSecundaria = cv2.resize(imgSecundaria, dsize=(
        250, 250), interpolation=cv2.INTER_CUBIC)

    imgSecundaria = Image.fromarray(imgSecundaria)
    imgSecundaria = ImageTk.PhotoImage(image=imgSecundaria)

    lblImagenDeseadaSecundaria.configure(image=imgSecundaria)


def imprimirImagenResultado(imagenResultado):
    global imgResult

    imagenResultado = Image.fromarray(imagenResultado)
    imgResult = ImageTk.PhotoImage(image=imagenResultado)
    lblImagenResultado.configure(image=imgResult)


def redimensionarImagen(imagen):
    imagen = cv2.resize(imagen, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    return imagen


def binarizarImagen():

    if(dirImgPrincipal != ''):

        umbral = tkinter.simpledialog.askinteger(
            'Ingresa valor', 'Ingresa un valor de umbralado')

        imgOriginal = cv2.imread(dirImgPrincipal, cv2.IMREAD_GRAYSCALE)

        ret, imgBinaria = cv2.threshold(
            imgOriginal, float(umbral), 255, cv2.THRESH_BINARY)

        imgRedimensionada = redimensionarImagen(imgBinaria)

        imprimirImagenResultado(imgRedimensionada)
    else:
        tkinter.messagebox.showwarning(
            message="Ocurrió un error al cargar la imagen principal", title="Error")


def generarHistograma():
    img_bgr = cv2.imread(dirImgPrincipal)

    color = ('b', 'g', 'r')

    for i, col in enumerate(color):

        histr = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        media = "Media de " + str(col) + ":" + str(np.mean(histr)) + "\n"
        moda = "Moda de " + str(col) + ":" + str(stats.mode(histr)[0]) + "\n"
        mediana = "Mediana de " + str(col) + ":" + str(np.median(histr)) + "\n"
        varianza = "Varianza de " + str(col) + ":" + str(np.var(histr)) + "\n"
        desviacion = "Desviacion estandar de " + \
            str(col) + ":" + str(np.std(histr)) + "\n"

        messagebox.showinfo(message=media + moda + mediana +
                            varianza + desviacion, title="Histograma por color")

        plt.plot(histr, color=col)
        plt.xlim([0, 256])
        plt.show()


def negarImagen():
    imgOriginal = cv2.imread(dirImgPrincipal)
    imgNegada = 255 - imgOriginal

    imgRedim = redimensionarImagen(imgNegada)
    imprimirImagenResultado(imgRedim)


def operacionPotencia():

    image = cv2.imread(dirImgPrincipal, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)

    gamma = tkinter.simpledialog.askfloat(
        'Ingresa valor', 'Ingresa el valor de gamma para potencia: ')

    imagePoten = image / 255
    powed = cv2.pow(imagePoten, float(gamma))

    cv2.imshow("Potencia", powed)

    resultRedim = redimensionarImagen(powed)
    imprimirImagenResultado(resultRedim)


def operacionLogaritmo():

    image = cv2.imread(dirImgPrincipal, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)

    c = tkinter.simpledialog.askfloat(
        'Ingresa valor', 'Ingresa el valor de c para logaritmo: ')

    log_image = float(c) * (np.log(image + 1.0))
    log_image = np.uint8(log_image + 0.5)

    cv2.imshow("Imagen Logaritmo", log_image)

    resultRedim = redimensionarImagen(log_image)
    imprimirImagenResultado(resultRedim)


def operacionPromedio():
    img = cv2.imread(dirImgPrincipal, 0)

    m, n = img.shape

    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9

    img_new = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0] + img[i, j] * \
                mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2,
                                                                         0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]

            img_new[i, j] = temp

    img_new = img_new.astype(np.uint8)

    resultRedim = redimensionarImagen(img_new)
    imprimirImagenResultado(resultRedim)


def operacionMediana():
    img_n1 = cv2.imread(dirImgPrincipal, 0)

    m, n = img_n1.shape

    img_new1 = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img_n1[i-1, j-1],
                    img_n1[i-1, j],
                    img_n1[i-1, j + 1],
                    img_n1[i, j-1],
                    img_n1[i, j],
                    img_n1[i, j + 1],
                    img_n1[i + 1, j-1],
                    img_n1[i + 1, j],
                    img_n1[i + 1, j + 1]]

            temp = sorted(temp)
            img_new1[i, j] = temp[4]

    img_new1 = img_new1.astype(np.uint8)

    resultRedim = redimensionarImagen(img_new1)
    imprimirImagenResultado(resultRedim)


def operacionLaplaciano():
    img = cv2.imread(dirImgPrincipal, 0).astype(np.float)
    K_size = 3

    H, W = img.shape

    pad = K_size // 2

    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)

    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    tmp = out.copy()

    K = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]

    for y in range(H):

        for x in range(W):

            out[pad + y, pad + x] = (-1) * np.sum(
                K * (tmp[y: y + K_size, x: x + K_size])) + tmp[pad + y, pad + x]

    out = np.clip(out, 0, 255)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    resultRedim = redimensionarImagen(out)
    imprimirImagenResultado(resultRedim)


def operacionSobel():
    img = cv2.imread(dirImgPrincipal, 0)
    threshold = 70

    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i+3, j:j+3]))  # vertical
            h = sum(sum(G_y * img[i:i+3, j:j+3]))  # horizon
            mag[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))

    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0

    resultRedim = redimensionarImagen(mag)
    imprimirImagenResultado(resultRedim)


def operacionSuma():

    image1 = cv2.imread(dirImgPrincipal)
    image2 = cv2.imread(dirImgSecundaria)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.add(image1, image2)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionResta():

    image1 = cv2.imread(dirImgPrincipal)
    image2 = cv2.imread(dirImgSecundaria)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.subtract(image1, image2)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionMultiplicar():

    image1 = cv2.imread(dirImgPrincipal)
    image2 = cv2.imread(dirImgSecundaria)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.multiply(image1, image2)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionDividir():

    image1 = cv2.imread(dirImgPrincipal)
    image2 = cv2.imread(dirImgSecundaria)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.divide(image1, image2)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionAND():
    image1 = cv2.imread(dirImgPrincipal)
    image2 = cv2.imread(dirImgSecundaria)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.bitwise_and(image1, image2)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionOR():
    image1 = cv2.imread(dirImgPrincipal)
    image2 = cv2.imread(dirImgSecundaria)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.bitwise_or(image1, image2)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionXOR():
    image1 = cv2.imread(dirImgPrincipal)
    image2 = cv2.imread(dirImgSecundaria)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.bitwise_xor(image1, image2)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionXOR():
    image1 = cv2.imread(dirImgPrincipal)

    image1 = cv2.resize(image1, dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC)

    result = cv2.bitwise_not(image1)

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def convertir_RGB_to_CMY():

    img = cv2.imread(dirImgPrincipal)
    img = cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)

    b, g, r = cv2.split(img)

    C = 1 - b/255
    M = 1 - g/255
    Y = 1 - r/255

    cv2.imshow("Imagen Original RGB", img)
    cv2.imshow("C", C)
    cv2.imshow("M", M)
    cv2.imshow("Y", Y)


def convertir_RGB_to_HSI():
    img = cv2.imread(dirImgPrincipal)

    img = cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)

    with np.errstate(divide='ignore', invalid='ignore'):

        bgr = np.float32(img)/255

        blue = bgr[:, :, 0]
        green = bgr[:, :, 1]
        red = bgr[:, :, 2]

        def calc_intensity(red, blue, green):
            return np.divide(blue + green + red, 3)

        def calc_saturation(red, blue, green):
            minimum = np.minimum(np.minimum(red, green), blue)
            saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

            return saturation

        def calc_hue(red, blue, green):
            hue = np.copy(red)

            for i in range(0, blue.shape[0]):
                for j in range(0, blue.shape[1]):
                    hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                        math.sqrt((red[i][j] - green[i][j])**2 +
                                  ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                    hue[i][j] = math.acos(hue[i][j])

                    if blue[i][j] <= green[i][j]:
                        hue[i][j] = hue[i][j]
                    else:
                        hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

            return hue

        hsi = cv2.merge((calc_hue(red, blue, green), calc_saturation(
            red, blue, green), calc_intensity(red, blue, green)))

        cv2.imshow("H", calc_hue(red, blue, green))
        cv2.imshow("S", calc_saturation(red, blue, green))
        cv2.imshow("I", calc_intensity(red, blue, green))
        cv2.imshow("HSI", hsi)


def operacionErosion():
    img = cv2.imread(dirImgPrincipal, 0)

    opcionKernel = tkinter.simpledialog.askfloat(
        'Ingresa valor', 'Que tipo de kernel deseas:\n1) MORPH_RECT\n2) MORPH_ELLIPSE\n3) MORPH_CROSS')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 50))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 30))

    if(opcionKernel == 1):
        result = cv2.erode(img, kernel)
    elif(opcionKernel == 2):
        result = cv2.erode(img, kernel1)
    elif(opcionKernel == 3):
        result = cv2.erode(img, kernel2)
    else:
        tkinter.messagebox.showwarning(
            message="Ocurrió un error al cargar la imagen principal", title="Error")

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionDilatacion():
    img = cv2.imread(dirImgPrincipal, 0)

    opcionKernel = tkinter.simpledialog.askfloat(
        'Ingresa valor', 'Que tipo de kernel deseas:\n1) MORPH_RECT\n2) MORPH_ELLIPSE\n3) MORPH_CROSS')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 50))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 30))

    if(opcionKernel == 1):
        result = cv2.dilate(img, kernel)
    elif(opcionKernel == 2):
        result = cv2.dilate(img, kernel1)
    elif(opcionKernel == 3):
        result = cv2.dilate(img, kernel2)
    else:
        tkinter.messagebox.showwarning(
            message="Ocurrió un error al cargar la imagen principal", title="Error")

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionApertura():
    img = cv2.imread(dirImgPrincipal, 0)

    opcionKernel = tkinter.simpledialog.askfloat(
        'Ingresa valor', 'Que tipo de kernel deseas:\n1) MORPH_RECT\n2) MORPH_ELLIPSE\n3) MORPH_CROSS')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 50))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 30))

    if(opcionKernel == 1):
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif(opcionKernel == 2):
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
    elif(opcionKernel == 3):
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    else:
        tkinter.messagebox.showwarning(
            message="Ocurrió un error al cargar la imagen principal", title="Error")

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


def operacionClausura():
    img = cv2.imread(dirImgPrincipal, 0)

    opcionKernel = tkinter.simpledialog.askfloat(
        'Ingresa valor', 'Que tipo de kernel deseas:\n1) MORPH_RECT\n2) MORPH_ELLIPSE\n3) MORPH_CROSS')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 50))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 30))

    if(opcionKernel == 1):
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif(opcionKernel == 2):
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    elif(opcionKernel == 3):
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    else:
        tkinter.messagebox.showwarning(
            message="Ocurrió un error al cargar la imagen principal", title="Error")

    resultRedim = redimensionarImagen(result)
    imprimirImagenResultado(resultRedim)


lblTituloPrograma = Label(view, text="MiniPhotoshop\nAutor: Ulises Miranda")

lblImagenDeseada = Label(view, text="Inserte Imagen")
lblImagenResultado = Label(view, text="Resultado de transformacion")

lblImagenDeseadaSecundaria = Label(view, text="Inserte Imagen secundaria")

btnSeleccionarImg = Button(
    view, text="Selecciona tu imagen 1", width=20, height=2, command=abrir_archivo)
btnSeleccionarImg2 = Button(
    view, text="Selecciona tu imagen 2", width=20, height=2, command=abrir_archivo2)

btnCambiarRGB_HSI = Button(view, text="Cambiar \nRGB->HSI",
                           width=20, height=2, command=convertir_RGB_to_HSI)
btnBinarizar = Button(view, text="Binarizar", width=20,
                      height=2, command=binarizarImagen)
btnHistograma = Button(view, text="Generar Histograma",
                       width=20, height=2, command=generarHistograma)
btnNegarImg = Button(view, text="Negar Imagen", width=20,
                     height=2, command=negarImagen)

btnCambiarRGB_CMY = Button(view, text="Cambiar\nRGB->CMY",
                           width=20, height=2, command=convertir_RGB_to_CMY)
btnOpLogaritmo = Button(view, text="Operacion log",
                        width=20, height=2, command=operacionLogaritmo)
btnOpPotencia = Button(view, text="Operacion Potencia",
                       width=20, height=2, command=operacionPotencia)
btnFilProm = Button(view, text="Filtro Promedio", width=20,
                    height=2, command=operacionPromedio)

btnFilMediana = Button(view, text="Filtro Mediana",
                       width=20, height=2, command=operacionMediana)
btnFilLapl = Button(view, text="Filtro Laplaciano", width=20,
                    height=2, command=operacionLaplaciano)
btnFilSobel = Button(view, text="Filtro Sobel", width=20,
                     height=2, command=operacionSobel)


btnSumarImagenes = Button(view, text="Sumar Imagenes",
                          width=20, height=2, command=operacionSuma)
btnRestarImagenes = Button(
    view, text="Restar Imagenes", width=20, height=2, command=operacionResta)
btnMultiplicarImagenes = Button(
    view, text="Multiplicar Imagenes", width=20, height=2, command=operacionMultiplicar)
btnDividirImagenes = Button(
    view, text="Dividir Imagenes", width=20, height=2, command=operacionDividir)

btnAND = Button(view, text="AND", width=20, height=2, command=operacionAND)
btnOR = Button(view, text="OR", width=20, height=2, command=operacionOR)
btnXOR = Button(view, text="XOR", width=20, height=2, command=operacionXOR)
btnNOT = Button(view, text="NOT", width=20, height=2, command=operacionAND)

btnErosion = Button(view, text="Erosion", width=20,
                    height=2, command=operacionErosion)
btnDilatacion = Button(view, text="Dilatacion", width=20,
                       height=2, command=operacionDilatacion)
btnApertura = Button(view, text="Apertura", width=20,
                     height=2, command=operacionApertura)
btnClausura = Button(view, text="Clausura", width=20,
                     height=2, command=operacionClausura)


lblTituloPrograma.grid(row=0, columnspan=2)
btnSeleccionarImg.grid(row=0, column=2)
btnSeleccionarImg2.grid(row=0, column=3)

btnCambiarRGB_CMY.grid(row=1, column=0)
btnBinarizar.grid(row=1, column=1)
btnHistograma.grid(row=1, column=2)
btnNegarImg.grid(row=1, column=3)
btnSumarImagenes.grid(row=1, column=4)
btnAND.grid(row=1, column=5)

btnCambiarRGB_HSI.grid(row=2, column=0)
btnOpLogaritmo.grid(row=2, column=1, columnspan=1)
btnOpPotencia.grid(row=2, column=2, columnspan=1)
btnRestarImagenes.grid(row=2, column=4)
btnOR.grid(row=2, column=5)

btnFilProm.grid(row=3, column=0)
btnFilMediana.grid(row=3, column=1)
btnFilLapl.grid(row=3, column=2)
btnFilSobel.grid(row=3, column=3)
btnMultiplicarImagenes.grid(row=3, column=4)
btnXOR.grid(row=3, column=5)

btnErosion.grid(row=4, column=0)
btnDilatacion.grid(row=4, column=1)
btnApertura.grid(row=4, column=2)
btnClausura.grid(row=4, column=3)
btnDividirImagenes.grid(row=4, column=4)
btnNOT.grid(row=4, column=5)

lblImagenDeseada.grid(row=5, column=0, columnspan=2)
lblImagenResultado.grid(row=5, column=2, columnspan=2)
lblImagenDeseadaSecundaria.grid(row=6, column=0, columnspan=2)

view.mainloop()
