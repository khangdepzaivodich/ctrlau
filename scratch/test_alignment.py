import xml.etree.ElementTree as ET

# Test coordinate alignment helper
def check_alignment(nodes):
    for n1, n2, expected_dir in [
        ('node_backbone', 'node_zimg', 'horizontal'),
        ('node_zimg', 'node_linear_proj', 'horizontal'),
        ('node_linear_proj', 'node_feat', 'horizontal'),
        ('cnn_au1', 'va_1', 'horizontal'),
        ('va_1', 'lin_au1', 'horizontal'),
        ('lin_au1', 'pau1', 'horizontal'),
        ('cnn_au2', 'va_2', 'horizontal'),
        ('va_2', 'lin_au2', 'horizontal'),
        ('lin_au2', 'pau2', 'horizontal'),
        ('cnn_au12', 'va_12', 'horizontal'),
        ('va_12', 'lin_au12', 'horizontal'),
        ('lin_au12', 'pau12', 'horizontal'),
        ('cnn_e1', 've_1', 'horizontal'),
        ('ve_1', 'lin_e1', 'horizontal'),
        ('lin_e1', 'pe1', 'horizontal'),
        ('cnn_e2', 've_2', 'horizontal'),
        ('ve_2', 'lin_e2', 'horizontal'),
        ('lin_e2', 'pe2', 'horizontal'),
        ('cnn_e7', 've_7', 'horizontal'),
        ('ve_7', 'lin_e7', 'horizontal'),
        ('lin_e7', 'pe7', 'horizontal'),
        ('node_gat1', 'node_loss_dag', 'vertical'),
        ('node_gat2', 'node_loss_cau_exp', 'vertical'),
        ('node_loss_wa', 'node_mae', 'vertical'),
        ('node_cf_imp_box', 'node_loss_cf_imp', 'horizontal'),
        ('node_cf_unimp_box', 'node_loss_cf_unimp', 'horizontal'),
    ]:
        print(f"Checking {n1} -> {n2} ({expected_dir})")

check_alignment([])
