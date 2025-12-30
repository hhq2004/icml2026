from abc import ABC, abstractmethod

class BaseMethod(ABC):
    def __init__(self, args, model):
        """
        args: 包含 budget, hyperparameters 等
        model: 加载好的 Backbone 模型 (因为有的方法需要修改模型内部，如 FastV)
        """
        self.args = args
        self.model = model
        self.token_budget = args.token_budget
        # Q-Frame等方法需要的温度参数
        self.temperature = getattr(args, 'temperature', 1.0)

    @abstractmethod
    def process_and_inference(self, video_path, question, options):
        """
        核心接口：
        1. 接收视频和问题
        2. 执行该方法特有的压缩/选择/记忆逻辑
        3. 调用模型进行推理
        4. 返回预测结果 (Choice index or Text)
        """
        pass