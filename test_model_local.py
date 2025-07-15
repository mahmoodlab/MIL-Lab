from src.builder import create_model

model= create_model('abmil.base.conch_v15.pc108-24k', from_pretrained=True, num_classes=5)

print("HIHIHIHIHIHI\n")

model2 = create_model('abmil.base.uni.pc108-24k', num_classes=5)

print("HIHIHIHIHIHI\n")

model3= create_model('abmil.base.uni_v2.pc108-24k', from_pretrained=True, num_classes=5)