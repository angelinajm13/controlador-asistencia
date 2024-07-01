import numpy
import cv2
import dlib
import face_recognition as fr

# cargar imagenes
foto_control = fr.load_image_file('Empleados/Angie.jpg')
foto_prueba = fr.load_image_file('2.jpg')

# pasar imagenes a rgb
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

# localizar cara control
lugar_cara_1 = fr.face_locations(foto_control)[0]
cara_codificada_A = fr.face_encodings(foto_control)[0]

# localizar cara prueba
lugar_cara_2 = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]


# mostrar rectangulo
cv2.rectangle(foto_control,
              (lugar_cara_1[3], lugar_cara_1[0]),
              (lugar_cara_1[1], lugar_cara_1[2]),
              (0, 255, 0),
              2)


# mostrar rectangulo
cv2.rectangle(foto_prueba,
              (lugar_cara_2[3], lugar_cara_2[0]),
              (lugar_cara_2[1], lugar_cara_2[2]),
              (0, 255, 0),
              2)

# realizar comparacion
resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B)


# medida de la distancia
distancia = fr.face_distance([cara_codificada_A], cara_codificada_B)

# mostrar resultado
cv2.putText(foto_prueba,
            f'{resultado} {distancia.round(2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2)

# mostrar imagenes
cv2.imshow('Foto Control', foto_control)
cv2.imshow('Foto Prueba', foto_prueba)

# mantener el programa abierto
cv2.waitKey(0)