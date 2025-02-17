
#####################################################

import streamlit as st  
import requests  
import json 
from PIL import Image  
from io import BytesIO  
from langchain.llms import HuggingFaceHub 
from langchain.chains import LLMChain 
from langchain.prompts import PromptTemplate 
from huggingface_hub import InferenceClient 
import time 
from openai import OpenAI 

#####################################################

# Exibição do titulo simulador HTML om CSS
image_url = "https://i.postimg.cc/s1FSW3dP/capa-dl.png"

# Define a largura desejada (em pixels)
st.image(image_url, width=710)

st.write("---")

#####################################################

# Defina a chave da API Hugging Face
HUGGINGFACE_API_KEY = "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"

# URLs da API
API_URL_DESCRICAO_IMG = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
API_URL_GENERATE_IMG = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

# Funções auxiliares
@st.cache_data
def describe_image(image_file):
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/octet-stream"
    }
    image_bytes = image_file.read()
    response = requests.post(API_URL_DESCRICAO_IMG, headers=headers, data=image_bytes)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
    return None

def generate_image(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(API_URL_GENERATE_IMG, headers=headers, json=payload)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    return None

def generate_postly(area, descricao):
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            content = f"Criar um texto para um anúncio na área '{area}' e baseado na descrição da imagem '{descricao}', para publicação em rede social."
            client = OpenAI(base_url="https://huggingface.co/api/inference-proxy/together", api_key=HUGGINGFACE_API_KEY)
            messages = [{"role": "user", "content": content}]
            completion = client.chat.completions.create(model="deepseek-ai/DeepSeek-V3", messages=messages, max_tokens=500)
            return completion.choices[0].message.content
        except Exception as e:
            st.warning(f"Tentativa {attempt + 1} falhou. Retentando...")
            time.sleep(5)
    return "Não foi possível gerar o anúncio. Verifique a API ou tente novamente."

def main():
    # Definindo as variáveis de sessão
    if 'rede_social' not in st.session_state:
        st.session_state.rede_social = None
    if 'contato' not in st.session_state:
        st.session_state.contato = None

    # Passo 1: Tipo de input
    tipo_input = st.radio("Selecione o tipo de input:", ("Imagem", "Texto"))

    if tipo_input:
        # Passo 2: Seleção de área
        area = st.selectbox("Defina uma área:", ["Comida", "Esporte", "Viagem", "Vestuário"])

    # st.write ("Selecioine a rede social foco da postagem:")
    st.markdown("<p style='font-size: 11px;color: #ffffff;'><p> Selecione a rede social foco da postagem:</p></p>", unsafe_allow_html=True)
    
    # Adicionando CSS personalizado para estilização dos botões
    st.markdown("""
        <style>
            .button-normal, .button-selected {
                color: white; /* Cor padrão */
                border: 1px solid white; /* Borda padrão */
                border-radius: 5px;
                padding: 5px 10px; /* Tamanho consistente */
                text-align: center;
                font-weight: normal; /* Sem negrito */
                display: inline-block; /* Consistência de layout */
                line-height: 1; /* Garantia de altura uniforme */
                box-sizing: border-box; /* Inclui borda no tamanho total */
            }
            .button-selected {
                color: #E91313; /* Fonte vermelha quando selecionado */
                border: 1px solid #E91313; /* Borda vermelha quando selecionado */
            }
        </style>
    """, unsafe_allow_html=True)
        
    # Redes sociais em uma única linha
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png", width=50)
        if st.session_state.rede_social == "LinkedIn":
            st.markdown('<div class="button-selected">LinkedIn</div>', unsafe_allow_html=True)
        else:
            if st.button('LinkedIn', key="linkedin"):
                st.session_state.rede_social = "LinkedIn"

    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png", width=50)
        if st.session_state.rede_social == "Instagram":
            st.markdown('<div class="button-selected">Instagram</div>', unsafe_allow_html=True)
        else:
            if st.button('Instagram', key="instagram"):
                st.session_state.rede_social = "Instagram"

    with col3:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg", width=50)
        if st.session_state.rede_social == "Facebook":
            st.markdown('<div class="button-selected">Facebook</div>', unsafe_allow_html=True)
        else:
            if st.button('Facebook', key="facebook"):
                st.session_state.rede_social = "Facebook"

    with col4:
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/WhatsApp.svg", width=50)
        if st.session_state.rede_social == "WhatsApp":
            st.markdown('<div class="button-selected">WhatsApp</div>', unsafe_allow_html=True)
        else:
            if st.button('WhatsApp', key="whatsapp"):
                st.session_state.rede_social = "WhatsApp"

    # Exibição do campo de contato
    if st.session_state.rede_social == "WhatsApp":
        st.session_state.contato = st.text_input("Informe o número do telefone com DDI (exemplo: +55 81 999887766):")
    elif st.session_state.rede_social:
        st.session_state.contato = st.text_input("Informe o perfil do usuário (exemplo: @usuario):")

    if st.session_state.contato:
        st.write(f"Contato para {st.session_state.rede_social}: {st.session_state.contato}")

    # Passo 4: Upload da imagem ou texto
    imagem = None
    texto = None
    
    if tipo_input == "Imagem":
        imagem = st.file_uploader("Faça o upload de uma imagem da área selecionada:", type=["jpg", "jpeg", "png"])
    elif tipo_input == "Texto":
        texto = st.text_area("Insira aqui um texto relacionado à área selecionada:")

    # Botão para gerar o conteúdo
    if st.button("Iniciar") and area and st.session_state.contato and st.session_state.rede_social:
        if tipo_input == "Imagem" and imagem:
            descricao = describe_image(imagem)
            texto = descricao  # Neste exemplo, apenas usamos a descrição
        elif tipo_input == "Texto" and texto:
            imagem = generate_image(texto)

        anuncio = generate_postly(area, texto)

        if imagem:
            st.image(imagem, use_column_width=True)

        # Exibição do resultado final
        st.markdown(f"""<div style="background-color: #262730; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #E91313;">
            <h3 style="color: #ffffff;">Sugestão para postagem:</h3>
            <h2 style="color: #8373C8;">{anuncio}</h2>
        </div>""", unsafe_allow_html=True)

        st.success("Anúncio gerado com sucesso!")

         # Botão "Postar anúncio" aparece após a geração bem-sucedida
        if st.button("Postar Anúncio"):
            st.success("Anúncio postado com sucesso!")
    else:
        st.error("Não foi possível gerar o anúncio. Tente novamente.")

if __name__ == "__main__":
    main()
