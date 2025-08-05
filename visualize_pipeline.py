import graphviz

dot = graphviz.Digraph('UnifiedPipeline', format='png')

# Define node styles
process_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#ff99ff'}
data_style = {'shape': 'box', 'style': 'filled', 'fillcolor': '#ffffff'}
model_style = {'shape': 'box', 'style': 'bold,filled', 'fillcolor': '#dddddd'}
output_style = {'shape': 'box', 'style': 'filled', 'fillcolor': '#99ff99'}

# Stage 1: Input & Preprocessing
dot.attr(label='Giai đoạn 1: Đầu vào & Tiền xử lý')

dot.node('A1', '''Người dùng cung cấp:
- Prompt & Negative Prompt
- Ảnh tham chiếu (AnyStory)
- Ảnh & Mask Inpainting (ControlNet)''', **process_style)

dot.node('B1_Text', 'Mã hóa Prompt\n(CLIP & T5 Encoders)', **model_style)
dot.node('C1_Text', '[Dữ liệu] Text Embeddings', **data_style)

dot.node('B2_AnyStory', 'Mã hóa Điều kiện AnyStory\n(Redux, Ref, Router Embedders)', **model_style)
dot.node('C2_AnyStory', '[Dữ liệu] AnyStory Embeddings\n(Ref, Redux, Router)', **data_style)

dot.node('B3_ControlNet', 'Chuẩn bị Điều kiện Inpainting\n(VAE Encode ảnh đã che + Ghép Mask)', **model_style)
dot.node('C3_ControlNet', '[Dữ liệu] ControlNet Condition', **data_style)

dot.node('B4_Latent', 'Chuẩn bị Latent\n(Tạo nhiễu ngẫu nhiên)', **process_style)
dot.node('D1', '[Dữ liệu] Latent nhiễu ban đầu (x_t)', **data_style)

dot.edges([
    ('A1', 'B1_Text'),
    ('B1_Text', 'C1_Text'),
    ('A1', 'B2_AnyStory'),
    ('B2_AnyStory', 'C2_AnyStory'),
    ('A1', 'B3_ControlNet'),
    ('B3_ControlNet', 'C3_ControlNet'),
    ('B4_Latent', 'D1'),
])

# Stage 2: Denoising Loop
dot.attr(label='Giai đoạn 2: Vòng lặp Khử nhiễu Hợp nhất')

dot.node('D1_loop', 'Latent nhiễu (x_t)', **data_style)
dot.node('C3_ControlNet_loop', 'ControlNet Condition', **data_style)
dot.node('C1_Text_loop', 'Text Embeddings', **data_style)
dot.node('E1_ControlNet', 'FluxControlNetModel', **model_style)
dot.node('F1_Residuals', '[Dữ liệu] Tín hiệu Điều khiển\n(ControlNet Residuals)', **data_style)

dot.edge('D1_loop', 'E1_ControlNet')
dot.edge('C3_ControlNet_loop', 'E1_ControlNet')
dot.edge('C1_Text_loop', 'E1_ControlNet')
dot.edge('E1_ControlNet', 'F1_Residuals')

dot.node('D1_loop2', 'Latent nhiễu (x_t)', **data_style)
dot.node('C1_Text_loop2', 'Text Embeddings', **data_style)
dot.node('C2_AnyStory_loop', 'AnyStory Embeddings', **data_style)
dot.node('F1_Residuals_loop', 'Tín hiệu Điều khiển', **data_style)
dot.node('E2_Transformer', 'FluxTransformer2DModel (Hợp nhất)', **model_style)
dot.node('F2_NoisePred', 'Dự đoán nhiễu (noise_pred)', **data_style)

dot.edge('D1_loop2', 'E2_Transformer')
dot.edge('C1_Text_loop2', 'E2_Transformer')
dot.edge('C2_AnyStory_loop', 'E2_Transformer')
dot.edge('F1_Residuals_loop', 'E2_Transformer')
dot.edge('E2_Transformer', 'F2_NoisePred')

dot.node('G1', 'merged_block_forward', shape='diamond')
dot.node('G2', '1. Chạy logic\nAnyStory Attention', **process_style)
dot.node('G3', '2. Cộng tín hiệu\nControlNet Residual', **process_style)

dot.edge('E2_Transformer', 'G1', label='Gọi hàm')
dot.edge('G1', 'G2')
dot.edge('G2', 'G3')

dot.node('H_Scheduler', 'Scheduler.step()', **process_style)
dot.node('I_NextLatent', '[Dữ liệu] Latent ít nhiễu hơn (x_t-1)', **data_style)
dot.edge('F2_NoisePred', 'H_Scheduler')
dot.edge('H_Scheduler', 'I_NextLatent')
dot.edge('I_NextLatent', 'D1_loop', style='dashed')
dot.edge('I_NextLatent', 'D1_loop2', style='dashed')

dot.edge('C1_Text', 'C1_Text_loop', style='dashed')
dot.edge('C1_Text', 'C1_Text_loop2', style='dashed')
dot.edge('C2_AnyStory', 'C2_AnyStory_loop', style='dashed')
dot.edge('C3_ControlNet', 'C3_ControlNet_loop', style='dashed')
dot.edge('D1', 'D1_loop', style='dashed')
dot.edge('D1', 'D1_loop2', style='dashed')

# Stage 3: Output
dot.attr(label='Giai đoạn 3: Đầu ra')

dot.node('J_FinalLatent', '[Dữ liệu] Latent cuối cùng (sạch)', **data_style)
dot.node('K_VAE', 'VAE Decoder', **model_style)
dot.node('L_Result', '(Kết quả) Ảnh cuối cùng', **output_style)

dot.edge('I_NextLatent', 'J_FinalLatent', label='Sau N bước')
dot.edge('J_FinalLatent', 'K_VAE')
dot.edge('K_VAE', 'L_Result')

# Render the diagram
dot.render('unified_pipeline', cleanup=False)
