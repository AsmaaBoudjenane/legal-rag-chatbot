#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interface.streamlit_app import LegalRAGInterface

if __name__ == "__main__":
    app = LegalRAGInterface()
    app.run()

# ===== 19. scripts/run_gradio.py =====
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interface.gradio_app import GradioInterface

if __name__ == "__main__":
    interface = GradioInterface()
    interface.launch(share=True)
