import demo.studio_server as ss
print(f"Has _state: {hasattr(ss, '_state')}")
if hasattr(ss, '_state'):
    print(f"_state: {ss._state}")
else:
    print("_state not found")
