import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64

app = Flask(__name__)

# Constantes e configurações
MODEL_PATH = "modelo_cabelo.tflite"  # Seu modelo específico
# Defina as classes de curvatura de cabelo
TIPOS_CABELO =["cacheado", "crespo", "liso", "ondulado"]  # Ajuste conforme as classes do seu modelo

# Carregue o modelo TFLite
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Obtenha informações sobre entradas e saídas do modelo
def get_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

# Pré-processamento da imagem
def preprocess_image(image, input_details):
    # Obtenha o tamanho esperado pelo modelo
    input_shape = input_details[0]['shape']
    
    # Remova a dimensão de batch se necessário
    if len(input_shape) == 4:
        target_height, target_width = input_shape[1], input_shape[2]
    else:
        target_height, target_width = input_shape[0], input_shape[1]
    
    # Redimensione a imagem para o tamanho esperado pelo modelo
    image = image.resize((target_width, target_height))
    
    # Converta para array numpy e normalize
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize os valores (geralmente para [-1, 1] ou [0, 1])
    image_array = image_array / 255.0
    
    # Adicione dimensão de batch se necessário
    if len(input_shape) == 4 and len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Realizar inferência com o modelo
def inference(interpreter, input_data, input_details, output_details):
    # Atribua o tensor de entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Execute a inferência
    interpreter.invoke()
    
    # Obtenha a saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Pós-processamento dos resultados
def postprocess_results(output_data):
    # Encontre a classe com a maior probabilidade
    if output_data.ndim > 1:
        class_index = np.argmax(output_data[0])
        confidence = float(output_data[0][class_index])
    else:
        class_index = np.argmax(output_data)
        confidence = float(output_data[class_index])
    
    # Mapeie para o tipo de cabelo correspondente
    tipo_cabelo = TIPOS_CABELO[class_index] if class_index < len(TIPOS_CABELO) else f"Tipo {class_index}"
    
    return {
        "tipo_cabelo": tipo_cabelo,
        "indice_classe": int(class_index),
        "confianca": confidence,
        "todas_probabilidades": output_data.flatten().tolist()
    }

# Carregue o modelo e obtenha detalhes
try:
    interpreter = load_tflite_model(MODEL_PATH)
    input_details, output_details = get_model_details(interpreter)
    model_loaded = True
    print(f"Modelo carregado com sucesso: {MODEL_PATH}")
    print(f"Forma de entrada esperada: {input_details[0]['shape']}")
except Exception as e:
    model_loaded = False
    print(f"Erro ao carregar o modelo: {e}")

@app.route('/analisar', methods=['POST'])
def analisar_cabelo():
    if not model_loaded:
        return jsonify({'erro': 'Modelo não carregado corretamente'}), 500
    
    try:
        # Verifique se foi enviado um arquivo
        if 'imagem' in request.files:
            imagem = request.files['imagem']
            img = Image.open(imagem)
        # Ou se foi enviado como base64
        elif 'imagem_base64' in request.json:
            img_data = base64.b64decode(request.json['imagem_base64'])
            img = Image.open(io.BytesIO(img_data))
        else:
            return jsonify({'erro': 'Nenhuma imagem fornecida'}), 400
        
        # Converta para RGB se estiver em outro formato
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Pré-processe a imagem
        input_data = preprocess_image(img, input_details)
        
        # Realize a inferência
        output_data = inference(interpreter, input_data, input_details, output_details)
        
        # Pós-processe os resultados
        resultado = postprocess_results(output_data)
        
        return jsonify(resultado)
    
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/info-modelo', methods=['GET'])
def info_modelo():
    if not model_loaded:
        return jsonify({'erro': 'Modelo não carregado corretamente'}), 500
    
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype'].__name__
    output_shape = output_details[0]['shape']
    output_type = output_details[0]['dtype'].__name__
    
    return jsonify({
        'modelo': 'Classificação de Curvaturas de Cabelo',
        'formato_entrada': input_shape,
        'tipo_entrada': input_type,
        'formato_saida': output_shape,
        'tipo_saida': output_type,
        'classes': TIPOS_CABELO
    })

if __name__ == '__main__':
    # Modo debug apenas em desenvolvimento local
    import os
    is_prod = os.environ.get('RENDER', False)
    app.run(debug=not is_prod, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))