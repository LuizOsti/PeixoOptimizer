import pyautogui
import pytesseract
import time
import cv2
import numpy as np
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyscreeze

# --- CONFIGURAÇÕES ---
# Garanta que o caminho para o Tesseract está correto no seu sistema
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Região onde o texto com a descrição do item aparece após o clique
REGIAO_TEXTO_DIREITA = (1352, 140, 468, 510)

# Lista de imagens de ícones que o script deve procurar
IMAGENS_MODELO = [
    'icone_item.png', 'icone_item1.png', 'icone_item2.png', 
    'icone_item3.png', 'icone_item4.png', 'icone_item5.png',
    'icone_item6.png', 'icone_item7.png'
]

# Região principal onde os itens do inventário são exibidos
REGIAO_DOS_ITENS = (169, 127, 1146, 800) 

# --- CONFIGURAÇÕES DE COMPORTAMENTO ---
NIVEL_CONFIANCA = 0.85 # Nível de confiança para encontrar as imagens
SCROLL_AMOUNT = -700   # Quantidade de pixels a rolar para baixo (valor negativo)
SCROLL_PAUSE_TIME = 2  # Pausa em segundos após rolar, para a UI estabilizar
MAX_WORKERS = 8        # Número de threads para processamento OCR paralelo
DISTANCIA_MINIMA = 30  # Distância em pixels para considerar dois itens como distintos
ARQUIVO_SAIDA = 'inventario_final_refatorado.json'

# --- FUNÇÕES AUXILIARES (sem alterações) ---

def agrupar_posicoes_proximas(lista_posicoes, distancia_minima=DISTANCIA_MINIMA):
    """Agrupa posições muito próximas para evitar detectar o mesmo item várias vezes na mesma tela."""
    if not lista_posicoes:
        return []
    posicoes_unicas = []
    # Ordenar ajuda a ter um processamento mais consistente
    lista_posicoes.sort(key=lambda p: (p.top, p.left))
    for pos in lista_posicoes:
        centro_pos = pyautogui.center(pos)
        if not any(math.dist(pyautogui.center(unica), centro_pos) < distancia_minima for unica in posicoes_unicas):
            posicoes_unicas.append(pos)
    return posicoes_unicas

def worker_ocr(screenshot, item_info):
    """Processa uma imagem para extrair texto usando Tesseract OCR."""
    try:
        imagem_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Pré-processamento da imagem para melhorar a qualidade do OCR
        imagem_grande = cv2.resize(imagem_cv, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        imagem_cinza = cv2.cvtColor(imagem_grande, cv2.COLOR_BGR2GRAY)
        _, imagem_processada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)
        imagem_processada = cv2.bitwise_not(imagem_processada)
        
        config_tesseract = '--psm 6'
        texto_extraido = pytesseract.image_to_string(imagem_processada, lang='eng', config=config_tesseract)
        
        resultado = {
            'item_numero': item_info['numero'],
            'posicao_clicada': item_info['posicao'],
            'texto_extraido': texto_extraido.strip()
        }
        return resultado
    except Exception as e:
        print(f"Erro no worker OCR para o item {item_info.get('numero')}: {e}")
        return None


def main():
    print("Scanner com Memória Inteligente (versão refatorada) começará em 3 segundos...")
    time.sleep(3)
    start_time = time.time()

    # <<< MUDANÇA PRINCIPAL 1: A MEMÓRIA MESTRA >>>
    # Este conjunto armazenará o centro (x, y) de CADA item já clicado.
    # É a nossa memória de longo prazo para evitar reprocessamento.
    centros_de_itens_ja_clicados = set()
    
    resultados_finais = []
    item_contador_global = 0
    falhas_consecutivas = 0
    
    while True:
        print("\n" + "="*50)
        print("Fase 1: Escaneando a área de itens visível...")
        
        # Encontra todas as ocorrências de todos os modelos de ícone na tela atual
        posicoes_encontradas_na_tela = []
        for modelo in IMAGENS_MODELO:
            try:
                encontrados = list(pyautogui.locateAllOnScreen(
                    modelo,
                    confidence=NIVEL_CONFIANCA,
                    region=REGIAO_DOS_ITENS,
                    grayscale=True
                ))
                if encontrados:
                    posicoes_encontradas_na_tela.extend(encontrados)
            except (pyscreeze.ImageNotFoundException, FileNotFoundError):
                # Ignora se um modelo de imagem específico não for encontrado
                continue
        
        # Agrupa posições muito próximas para ter uma lista limpa de itens na tela
        posicoes_unicas_tela_atual = agrupar_posicoes_proximas(posicoes_encontradas_na_tela)
        print(f"Encontrados {len(posicoes_unicas_tela_atual)} itens únicos na tela atual.")

        # <<< MUDANÇA PRINCIPAL 2: O SISTEMA DE FILTRAGEM >>>
        # Filtra a lista de itens da tela atual, mantendo apenas aqueles
        # que ainda não foram clicados (comparando com a memória mestra).
        novas_posicoes_nesta_tela = []
        for pos in posicoes_unicas_tela_atual:
            centro_pos = pyautogui.center(pos)
            
            # Verifica se o centro do item encontrado está longe o suficiente de TODOS os itens já clicados
            e_realmente_novo = not any(
                math.dist(centro_pos, centro_ja_visto) < DISTANCIA_MINIMA 
                for centro_ja_visto in centros_de_itens_ja_clicados
            )
            
            if e_realmente_novo:
                novas_posicoes_nesta_tela.append(pos)
        
        # --- Lógica de Processamento e Parada ---
        if novas_posicoes_nesta_tela:
            falhas_consecutivas = 0 # Reseta o contador de falhas pois encontramos trabalho a fazer
            print(f"Fase 2: Processando {len(novas_posicoes_nesta_tela)} itens NOVOS encontrados...")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                tarefas_ocr = []
                for posicao in novas_posicoes_nesta_tela:
                    item_contador_global += 1
                    centro_do_item = pyautogui.center(posicao)
                    
                    # <<< MUDANÇA PRINCIPAL 3: ATUALIZANDO A MEMÓRIA >>>
                    # Adiciona o centro do item à memória mestra ANTES de processar
                    # para evitar que seja adicionado à fila novamente.
                    centros_de_itens_ja_clicados.add(centro_do_item)
                    
                    # Clica no item e tira o screenshot da descrição
                    pyautogui.click(centro_do_item, clicks=1, interval=0.01, duration=0.01)
                    time.sleep(0.05) # Pequena pausa para a UI reagir ao clique
                    
                    screenshot = pyautogui.screenshot(region=REGIAO_TEXTO_DIREITA)
                    item_info = {
                        'numero': item_contador_global,
                        'posicao': {'x': int(centro_do_item.x), 'y': int(centro_do_item.y)}
                    }
                    tarefas_ocr.append(executor.submit(worker_ocr, screenshot, item_info))
                
                # Coleta os resultados das tarefas OCR
                for futura_tarefa in as_completed(tarefas_ocr):
                    resultado = futura_tarefa.result()
                    if resultado:
                        resultados_finais.append(resultado)
            
            print(f"Leva de {len(novas_posicoes_nesta_tela)} itens processada. Total acumulado na memória: {len(centros_de_itens_ja_clicados)} itens.")
        
        else: # Nenhum item NOVO foi encontrado na tela visível
            falhas_consecutivas += 1
            print(f"Fase 2: Nenhum item novo encontrado nesta tela. (Falha consecutiva #{falhas_consecutivas})")

        # Condição de parada: se por 2 vezes seguidas não encontramos nada novo, assumimos que acabou.
        if falhas_consecutivas >= 2:
            print("\nNenhum item novo encontrado após múltiplas tentativas. Considerado fim do inventário.")
            break
            
        # <<< MUDANÇA PRINCIPAL 4: ROLAGEM ÚNICA POR CICLO >>>
        # Rola para baixo apenas UMA VEZ por ciclo do loop `while`.
        print("Fase 3: Rolando para a próxima seção do inventário...")
        pyautogui.scroll(SCROLL_AMOUNT)
        time.sleep(SCROLL_PAUSE_TIME)

    # --- Finalização ---
    end_time = time.time()
    resultados_finais.sort(key=lambda r: r['item_numero'])
    
    print("\n" + "="*50)
    print("PROCESSO CONCLUÍDO!")
    print(f"Tempo total de execução: {end_time - start_time:.2f} segundos.")
    print(f"Total de itens únicos escaneados: {len(resultados_finais)}")
    
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(resultados_finais, f, ensure_ascii=False, indent=4)
    print(f"Resultados salvos com sucesso em '{ARQUIVO_SAIDA}'")

if __name__ == "__main__":
    main()