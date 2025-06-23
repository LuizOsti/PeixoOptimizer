import pyautogui
import pytesseract
import time
import cv2
import numpy as np
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyscreeze
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
REGIAO_TEXTO_DIREITA = (1352, 140, 468, 510)

CAMINHO_DO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_TEMPLATES = os.path.join(CAMINHO_DO_SCRIPT, 'scans_templates')

NOMES_ARQUIVOS_MODELO = [
    'icone_item.png', 'icone_item1.png', 'icone_item2.png',
    'icone_item3.png', 'icone_item4.png', 'icone_item5.png',
    'icone_item6.png', 'icone_item7.png'
]
IMAGENS_MODELO = [os.path.join(PASTA_TEMPLATES, nome) for nome in NOMES_ARQUIVOS_MODELO]

MAPA_DE_SIMBOLOS = {
    "keeneye": "simbolo_keeneye.png",
    "swiftrush": "simbolo_swiftrush.png",
    "battlewill": "simbolo_battlewil.png",
    "bloodbath" : "simbolo_bloodbath.png",
    "bramble" : "simbolo_bramble.png",
    "bulwark" : "simbolo_bulwark.png",
    "cure" : "simbolo_cure.png",
    "fury" : "simbolo_fury.png",
    "harvest" : "simbolo_harvest.png",
    "momemtum" : "simbolo_momentum.png",
    "onslaught" : "simbolo_onslaught.png",
    "strive" : "simbolo_strive.png",
    "timewave" : "simbolo_timewave.png",
    "unbreakable" : "simbolo_unbreakable.png",
    "wellspring" : "simbolo_wellspring.png"

}

REGIAO_DOS_ITENS = (169, 127, 1146, 700)

pyautogui.PAUSE = 0
NIVEL_CONFIANCA_ITEM = 0.85
NIVEL_CONFIANCA_SIMBOLO = 0.90
MAX_WORKERS = 8
DISTANCIA_MINIMA_AGRUPAMENTO = 30
PAUSA_POS_CLIQUE = 0.1
SCROLL_AMOUNT = -3000
ARQUIVO_SAIDA = 'gear.json'

def agrupar_posicoes_proximas(lista_posicoes):
    if not lista_posicoes: return []
    posicoes_unicas = []
    lista_posicoes.sort(key=lambda p: (p.top, p.left))
    for pos in lista_posicoes:
        centro_pos = pyautogui.center(pos)
        if not any(math.dist(pyautogui.center(unica), centro_pos) < DISTANCIA_MINIMA_AGRUPAMENTO for unica in posicoes_unicas):
            posicoes_unicas.append(pos)
    return posicoes_unicas

def normalizar_texto(texto):
    return "".join(texto.lower().split())

def contar_simbolos(imagem_template, regiao_busca):
    try:
        caminho_completo = os.path.join(PASTA_TEMPLATES, imagem_template)
        ocorrencias = list(pyautogui.locateAllOnScreen(
            caminho_completo,
            region=regiao_busca,
            confidence=NIVEL_CONFIANCA_SIMBOLO
        ))
        return len(ocorrencias)
    except (pyscreeze.PyScreezeException, FileNotFoundError):
        return 0

def worker_ocr(screenshot):
    try:
        imagem_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        imagem_cinza = cv2.cvtColor(imagem_cv, cv2.COLOR_BGR2GRAY)
        fator_resize = 2.0
        imagem_grande = cv2.resize(imagem_cinza, None, fx=fator_resize, fy=fator_resize, interpolation=cv2.INTER_LANCZOS4)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        imagem_nitida = cv2.filter2D(imagem_grande, -1, kernel)
        imagem_processada = cv2.adaptiveThreshold(
            imagem_nitida, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        config_tesseract = '--psm 6 --oem 1'
        texto_extraido = pytesseract.image_to_string(imagem_processada, lang='eng', config=config_tesseract)
        return texto_extraido.strip()
    except Exception as e:
        return f"ERRO_OCR: {e}"

def main():
    print("Gear Scanner (v3: Mapeamento Dinâmico de Símbolos) iniciará em 3 segundos...")
    time.sleep(3)
    start_time = time.time()

    identificadores_de_texto_ja_vistos = set()
    resultados_finais = []
    item_contador_global = 0
    falhas_consecutivas = 0

    while True:
        print("\n--- Procurando ícones...")
        posicoes_encontradas_na_tela = []
        for modelo in IMAGENS_MODELO:
            try:
                encontrados = list(pyautogui.locateAllOnScreen(
                    modelo, confidence=NIVEL_CONFIANCA_ITEM, region=REGIAO_DOS_ITENS, grayscale=True
                ))
                posicoes_encontradas_na_tela.extend(encontrados)
            except pyscreeze.PyScreezeException: continue
        
        posicoes_para_clicar = agrupar_posicoes_proximas(posicoes_encontradas_na_tela)
        
        if not posicoes_para_clicar:
            print("Nenhum ícone de item encontrado neste ciclo.")
            falhas_consecutivas += 1
        else:
            print(f"Coletando {len(posicoes_para_clicar)} screenshots para OCR...")
            screenshots_para_processar = []
            for posicao in posicoes_para_clicar:
                pyautogui.click(pyautogui.center(posicao))
                time.sleep(PAUSA_POS_CLIQUE)
                screenshots_para_processar.append(pyautogui.screenshot(region=REGIAO_TEXTO_DIREITA))

            print("Processando OCR em paralelo...")
            resultados_ocr_da_tela = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                tarefas_ocr = [executor.submit(worker_ocr, screenshot) for screenshot in screenshots_para_processar]
                for futura_tarefa in as_completed(tarefas_ocr):
                    resultados_ocr_da_tela.append(futura_tarefa.result())

            print("Verificando resultados e contando símbolos...")
            itens_realmente_novos_nesta_leva = 0
            for texto_original in resultados_ocr_da_tela:
                if not texto_original or "ERRO_OCR" in texto_original: continue
                id_do_texto = normalizar_texto(texto_original)
                
                if id_do_texto not in identificadores_de_texto_ja_vistos:
                    itens_realmente_novos_nesta_leva += 1
                    item_contador_global += 1
                    identificadores_de_texto_ja_vistos.add(id_do_texto)
                    
                    tipo_simbolo_encontrado = "Nenhum"
                    contagem_simbolo = 0
                    
                    texto_lower = texto_original.lower()
                    for palavra_chave, arquivo_simbolo in MAPA_DE_SIMBOLOS.items():
                        if palavra_chave in texto_lower:
                            tipo_simbolo_encontrado = palavra_chave
                            contagem_simbolo = contar_simbolos(arquivo_simbolo, REGIAO_DOS_ITENS)
                            break
                    
                    item_formatado = {
                        'item_numero': item_contador_global,
                        'texto_extraido': texto_original,
                        'tipo_simbolo': tipo_simbolo_encontrado,
                        'contagem_simbolo': contagem_simbolo
                    }
                    resultados_finais.append(item_formatado)
            
            print(f"{itens_realmente_novos_nesta_leva} itens novos encontrados.")

            if itens_realmente_novos_nesta_leva == 0:
                falhas_consecutivas += 1
            else:
                falhas_consecutivas = 0
                print("Rolando para baixo...")
                pyautogui.scroll(SCROLL_AMOUNT)

        if falhas_consecutivas >= 2:
            print("\nFim do inventário.")
            break

    print("\n" + "="*50 + "\nPROCESSO CONCLUÍDO!")
    print(f"Total de {len(resultados_finais)} itens únicos foram lidos e salvos.")
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(resultados_finais, f, ensure_ascii=False, indent=4)
    print(f"Resultados salvos em '{ARQUIVO_SAIDA}'.")
    end_time = time.time()
    print(f"Tempo de execução total: {end_time - start_time:.2f} segundos.")

if __name__ == "__main__":
    main()