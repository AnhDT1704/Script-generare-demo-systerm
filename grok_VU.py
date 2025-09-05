import os
import shutil
import json
import hashlib
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import subprocess
import sys

from moviepy.editor import VideoFileClip
import google.generativeai as genai
import base64
import tempfile
import time
import subprocess
from pathlib import Path
import PIL.Image
import io
from openai import OpenAI
import re
import httpx  # Thêm import httpx cho OpenRouter client

@dataclass
class VideoInfo:
    id: str
    original_name: str
    stored_path: str
    compressed_path: Optional[str]
    duration: float
    width: int
    height: int
    file_size_mb: float
    upload_time: str
    analysis_result: Optional[str] = None
    status: str = "uploaded"  # uploaded, processing, analyzed, error

class OptimizedVideoAnalyzer:
    """
    Class phân tích video tối ưu với nhiều phương pháp backup
    """
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.temp_dir = tempfile.mkdtemp()
        
    def analyze_video_robust(self, video_path: str) -> str:
        """
        Phân tích video với nhiều phương pháp backup
        
        Args:
            video_path (str): Đường dẫn video
            
        Returns:
            str: Mô tả nội dung video
        """
        # Phương pháp 1: Phân tích video trực tiếp
        try:
            print(f"🎬 Phương pháp 1 - Upload video: {os.path.basename(video_path)}")
            return self.analyze_video_direct(video_path)
        except Exception as e:
            print(f"❌ Phương pháp 1 thất bại: {e}")
            
        # Phương pháp 2: Phân tích qua frames
        try:
            print(f"🖼️ Phương pháp 2 - Phân tích frames: {os.path.basename(video_path)}")
            return self.analyze_video_frames(video_path)
        except Exception as e:
            print(f"❌ Phương pháp 2 thất bại: {e}")
            
        # Phương pháp 3: Mô tả dựa trên metadata
        print(f"📊 Phương pháp 3 - Mô tả metadata: {os.path.basename(video_path)}")
        return self.analyze_video_metadata(video_path)
    
    def analyze_video_direct(self, video_path: str) -> str:
        """Phân tích video trực tiếp qua Gemini"""
        # Kiểm tra file size
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        if file_size > 100:
            raise Exception(f"File quá lớn ({file_size:.1f}MB), cần nén trước")
        
        # Upload với retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                video_file = genai.upload_file(path=video_path)
                print(f"✅ Upload thành công (lần {attempt + 1})")
                
                # Chờ xử lý
                time.sleep(3)
                
                prompt = """
                Phân tích video này và trả về CHỈ trong định dạng sau, không thêm bất kỳ nội dung nào khác:
                1. Bối cảnh, môi trường chính: [mô tả chi tiết]
                2. Nhân vật/đối tượng xuất hiện: [mô tả chi tiết]
                3. Hoạt động chính diễn ra: [mô tả cực kì chi tiết, đầy đủ thông tin]
                4. Cảm xúc, không khí: [mô tả chi tiết]
                """
                
                response = self.model.generate_content([video_file, prompt])
                
                # Cleanup
                video_file.delete()
                
                if response and response.text:
                    return response.text.strip()
                else:
                    raise Exception("Gemini trả về response trống")
                    
            except Exception as e:
                print(f"❌ Lần thử {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise e
    
    def analyze_video_frames(self, video_path: str) -> str:
        """Phân tích video qua frames"""
        video = VideoFileClip(video_path)
        total_frames = int(video.fps * video.duration)
        
        # Lấy 3 frames: đầu, giữa, cuối
        frame_positions = [0, total_frames//2, total_frames-1]
        images = []
        
        for pos in frame_positions:
            time = pos / video.fps
            frame = video.get_frame(time)
            # Resize frame to 640x480
            pil_image = PIL.Image.fromarray(frame).resize((640, 480))
            images.append(pil_image)
        
        video.close()
        
        if not images:
            raise Exception("Không thể trích xuất frames")
        
        prompt = f"""
        Dựa vào {len(images)} frames từ video, mô tả nội dung chính:
        Phân tích video này và trả về CHỈ trong định dạng sau, không thêm bất kỳ nội dung nào khác:
                1. Bối cảnh, môi trường chính: [mô tả chi tiết]
                2. Nhân vật/đối tượng xuất hiện: [mô tả chi tiết]
                3. Hoạt động chính diễn ra: [mô tả chi tiết]
                4. Cảm xúc, không khí: [mô tả chi tiết]
        """
        
        content = [prompt] + images
        response = self.model.generate_content(content)
        
        return response.text.strip() if response and response.text else "Nội dung video chưa xác định"
    
    def analyze_video_metadata(self, video_path: str) -> str:
        """Tạo mô tả dựa trên metadata"""
        try:
            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()
            
            # Dùng AI để sinh mô tả hợp lý
            prompt = f"""
            Dựa vào thông tin:
            - Video thời lượng: {duration:.1f} giây
            - Tên file: {os.path.basename(video_path)}
            
            Hãy tạo mô tả hợp lý cho cảnh video này.
            Ví dụ:
            - 5-15s: cảnh giới thiệu, logo, chào mừng
            - 15-30s: cảnh chuyển tiếp, hoạt động
            - 30s+: nội dung chính, biểu diễn, giao lưu
            
            Trả về mô tả chi tiết về nội dung video.
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip() if response and response.text else f"Cảnh video thời lượng {duration:.0f} giây"
            
        except Exception as e:
            return f"Cảnh video (metadata error: {e})"
    
    def cleanup(self):
        """Dọn dẹp thư mục tạm"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Lỗi dọn dẹp: {e}")

class VideoStorageManager:
    def __init__(self, storage_root: str = "./video_storage"):
        """
        Khởi tạo hệ thống lưu trữ video
        
        Args:
            storage_root (str): Thư mục gốc lưu trữ
        """
        self.storage_root = Path(storage_root)
        self.original_dir = self.storage_root / "originals"
        self.compressed_dir = self.storage_root / "compressed"
        self.db_path = self.storage_root / "video_database.db"
        
        # Tạo thư mục
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.compressed_dir.mkdir(parents=True, exist_ok=True)
        
        # Khởi tạo database
        self.init_database()
    
    def init_database(self):
        """Khởi tạo SQLite database để quản lý metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id TEXT PRIMARY KEY,
                original_name TEXT,
                stored_path TEXT,
                compressed_path TEXT,
                duration REAL,
                width INTEGER,
                height INTEGER,
                file_size_mb REAL,
                upload_time TEXT,
                analysis_result TEXT,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_video_id(self, file_path: str) -> str:
        """
        Tạo ID duy nhất cho video dựa trên hash
        
        Args:
            file_path (str): Đường dẫn file
            
        Returns:
            str: Video ID
        """
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash[:16]
    
    def extract_video_metadata(self, video_path: str) -> Dict:
        """
        Trích xuất metadata từ video
        
        Args:
            video_path (str): Đường dẫn video
            
        Returns:
            Dict: Metadata video
        """
        video = VideoFileClip(video_path)
        
        duration = video.duration
        width, height = video.size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        video.close()
        
        return {
            "duration": duration,
            "width": width,
            "height": height,
            "file_size_mb": file_size_mb
        }
    
    def store_video(self, video_path: str, original_name: str = None) -> str:
        """
        Lưu trữ video vào hệ thống
        
        Args:
            video_path (str): Đường dẫn video gốc
            original_name (str): Tên gốc của file
            
        Returns:
            str: Video ID
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video không tồn tại: {video_path}")
        
        # Tạo ID và tên file
        video_id = self.generate_video_id(video_path)
        if original_name is None:
            original_name = os.path.basename(video_path)
        
        # Kiểm tra đã tồn tại chưa
        if self.get_video_info(video_id):
            print(f"Video đã tồn tại: {video_id}")
            return video_id
        
        # Trích xuất metadata
        metadata = self.extract_video_metadata(video_path)
        
        # Copy file gốc
        file_extension = os.path.splitext(original_name)[1]
        stored_path = self.original_dir / f"{video_id}{file_extension}"
        shutil.copy2(video_path, stored_path)
        
        # Tạo VideoInfo
        video_info = VideoInfo(
            id=video_id,
            original_name=original_name,
            stored_path=str(stored_path),
            compressed_path=None,
            duration=metadata["duration"],
            width=metadata["width"],
            height=metadata["height"],
            file_size_mb=metadata["file_size_mb"],
            upload_time=datetime.now().isoformat(),
            status="uploaded"
        )
        
        # Lưu vào database
        self.save_video_info(video_info)
        
        print(f"✅ Đã lưu trữ video: {original_name} -> {video_id}")
        return video_id
    
    def compress_and_store(self, video_id: str) -> bool:
        """
        Nén và lưu trữ version nén của video
        
        Args:
            video_id (str): ID video
            
        Returns:
            bool: True nếu nén thành công
        """
        video_info = self.get_video_info(video_id)
        if not video_info:
            return False
        
        # Nếu đã có version nén
        if video_info.compressed_path and os.path.exists(video_info.compressed_path):
            return True

        # Nếu file nhỏ hơn 100MB, không cần nén
        if video_info.file_size_mb <= 100:
            video_info.compressed_path = video_info.stored_path
            self.save_video_info(video_info)
            return True
        
        # Nén video
        try:
            compressed_path = self.compressed_dir / f"{video_id}_compressed.mp4"
            
            if self._compress_video_opencv(video_info.stored_path, str(compressed_path)):
                video_info.compressed_path = str(compressed_path)
                video_info.status = "compressed"
                self.save_video_info(video_info)
                print(f"✅ Đã nén video {video_id}")
                return True
            else:
                print(f"⚠️ Không thể nén video {video_id}")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi nén video {video_id}: {e}")
            return False
    
    def _compress_video_opencv(self, input_path: str, output_path: str) -> bool:
        """Nén video bằng moviepy"""
        try:
            video = VideoFileClip(input_path)
            
            # Tối ưu thông số
            new_fps = min(video.fps, 24)
            new_width = min(video.size[0], 1280)
            new_height = min(video.size[1], 720)
            
            # Giữ tỷ lệ khung hình
            if video.size[0] > video.size[1]:
                new_height = int(new_width * video.size[1] / video.size[0])
            else:
                new_width = int(new_height * video.size[0] / video.size[1])
            
            # Resize và nén video
            resized_clip = video.resize(width=new_width)
            
            # Nén với bitrate thấp hơn để đạt kích thước mục tiêu (30MB)
            target_size_mb = 30
            duration = video.duration
            target_bitrate = str(int((target_size_mb * 8192) / duration)) + 'k'
            
            resized_clip.write_videofile(
                output_path,
                fps=new_fps,
                preset='veryslow',  # Nén chất lượng cao nhất có thể
                bitrate=target_bitrate,
                codec='libx264'
            )
            
            video.close()
            resized_clip.close()
            
            # Kiểm tra kết quả
            new_size = os.path.getsize(output_path) / (1024 * 1024)
            return new_size < 30
            
        except Exception as e:
            print(f"Lỗi nén OpenCV: {e}")
            return False
    
    def save_video_info(self, video_info: VideoInfo):
        """Lưu thông tin video vào database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO videos 
            (id, original_name, stored_path, compressed_path, duration, width, height, 
             file_size_mb, upload_time, analysis_result, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_info.id, video_info.original_name, video_info.stored_path,
            video_info.compressed_path, video_info.duration, video_info.width,
            video_info.height, video_info.file_size_mb, video_info.upload_time,
            video_info.analysis_result, video_info.status
        ))
        
        conn.commit()
        conn.close()
    
    def get_video_info(self, video_id: str) -> Optional[VideoInfo]:
        """Lấy thông tin video từ database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return VideoInfo(
                id=row[0], original_name=row[1], stored_path=row[2],
                compressed_path=row[3], duration=row[4], width=row[5],
                height=row[6], file_size_mb=row[7], upload_time=row[8],
                analysis_result=row[9], status=row[10]
            )
        return None
    
    def get_all_videos(self) -> List[VideoInfo]:
        """Lấy danh sách tất cả video đã lưu trữ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM videos ORDER BY upload_time DESC')
        rows = cursor.fetchall()
        conn.close()
        
        videos = []
        for row in rows:
            videos.append(VideoInfo(
                id=row[0], original_name=row[1], stored_path=row[2],
                compressed_path=row[3], duration=row[4], width=row[5],
                height=row[6], file_size_mb=row[7], upload_time=row[8],
                analysis_result=row[9], status=row[10]
            ))
        
        return videos
    
    def get_video_for_analysis(self, video_id: str) -> Optional[str]:
        """
        Lấy đường dẫn video tối ưu để phân tích
        
        Args:
            video_id (str): ID video
            
        Returns:
            Optional[str]: Đường dẫn video (ưu tiên compressed nếu có)
        """
        video_info = self.get_video_info(video_id)
        if not video_info:
            return None
        
        # Ưu tiên file nén nếu có
        if video_info.compressed_path and os.path.exists(video_info.compressed_path):
            return video_info.compressed_path
        
        # Fallback về file gốc
        if os.path.exists(video_info.stored_path):
            return video_info.stored_path
        
        return None
    
    def update_analysis_result(self, video_id: str, analysis_result: str):
        """Cập nhật kết quả phân tích"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE videos 
            SET analysis_result = ?, status = 'analyzed'
            WHERE id = ?
        ''', (analysis_result, video_id))
        
        conn.commit()
        conn.close()
    
    def delete_video(self, video_id: str) -> bool:
        """
        Xóa video khỏi hệ thống lưu trữ
        
        Args:
            video_id (str): ID video cần xóa
            
        Returns:
            bool: True nếu xóa thành công
        """
        try:
            video_info = self.get_video_info(video_id)
            if not video_info:
                return False
            
            # Xóa files
            if os.path.exists(video_info.stored_path):
                os.remove(video_info.stored_path)
            
            if video_info.compressed_path and os.path.exists(video_info.compressed_path):
                os.remove(video_info.compressed_path)
            
            # Xóa record trong database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM videos WHERE id = ?', (video_id,))
            conn.commit()
            conn.close()
            
            print(f"✅ Đã xóa video: {video_info.original_name}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi xóa video {video_id}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict:
        """Lấy thống kê hệ thống lưu trữ"""
        videos = self.get_all_videos()
        
        total_videos = len(videos)
        total_size = sum(v.file_size_mb for v in videos)
        analyzed_videos = len([v for v in videos if v.status == "analyzed"])
        
        return {
            "total_videos": total_videos,
            "total_size_mb": total_size,
            "analyzed_videos": analyzed_videos,
            "storage_path": str(self.storage_root)
        }
    
    def cleanup_old_videos(self, days_old: int = 30):
        """
        Dọn dẹp video cũ
        
        Args:
            days_old (int): Xóa video cũ hơn số ngày này
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        videos = self.get_all_videos()
        
        deleted_count = 0
        for video in videos:
            upload_time = datetime.fromisoformat(video.upload_time)
            if upload_time < cutoff_date:
                if self.delete_video(video.id):
                    deleted_count += 1
        
        print(f"🧹 Đã dọn dẹp {deleted_count} video cũ")

class EnhancedVideoScriptGenerator:
    def __init__(self, api_key: str, claude_api_key: str, use_openrouter: bool = True, storage_root: str = "./video_storage"):
        """
        Hệ thống tạo kịch bản nâng cao với storage và Claude cho voiceover.
        
        Args:
            api_key (str): Gemini API key
            claude_api_key (str): OpenRouter API key
            use_openrouter (bool): True để dùng OpenRouter
            storage_root (str): Thư mục lưu trữ
        """
        self.storage = VideoStorageManager(storage_root)
        self.analyzer = OptimizedVideoAnalyzer(api_key)  # Gemini cho analyze video
        self.claude_api_key = claude_api_key
        self.use_openrouter = use_openrouter
        
        if self.use_openrouter:
            # Cấu hình cho OpenRouter với OpenAI client phiên bản mới
            from openai import AsyncOpenAI, OpenAI
            import httpx
            
            self.claude_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.claude_api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/AnhDT1704/Script-generare-demo-systerm",
                    "X-Title": "Video Analysis & Script Generator",
                    "User-Agent": "Script Generator Demo v1.0"
                },
                timeout=60.0  # Tăng timeout để tránh lỗi timeout
            )
        else:
            from anthropic import Anthropic
            self.claude_client = Anthropic(api_key=self.claude_api_key)
    
    def process_video_uploads(self, uploaded_files: List[str]) -> List[str]:
        """
        Xử lý danh sách video upload
        
        Args:
            uploaded_files (List[str]): Danh sách đường dẫn video
            
        Returns:
            List[str]: Danh sách video IDs
        """
        video_ids = []
        
        print("📁 Đang lưu trữ videos...")
        for video_path in uploaded_files:
            try:
                video_id = self.storage.store_video(video_path)
                
                # Nén video nếu cần
                self.storage.compress_and_store(video_id)
                
                video_ids.append(video_id)
                print(f"✅ Đã lưu trữ: {os.path.basename(video_path)} -> {video_id}")
                
            except Exception as e:
                print(f"❌ Lỗi lưu trữ {os.path.basename(video_path)}: {e}")
        
        return video_ids
    
    def analyze_stored_videos(self, video_ids: List[str]) -> Dict[str, str]:
        """
        Phân tích các video đã lưu trữ
        
        Args:
            video_ids (List[str]): Danh sách video IDs
            
        Returns:
            Dict[str, str]: Map từ video_id đến analysis result
        """
        results = {}
        
        print("🤖 Đang phân tích nội dung videos...")
        
        for video_id in video_ids:
            try:
                # Lấy đường dẫn video tối ưu
                video_path = self.storage.get_video_for_analysis(video_id)
                if not video_path:
                    continue
                
                # Phân tích nội dung
                analysis_result = self.analyzer.analyze_video_robust(video_path)
                
                # Lưu kết quả
                self.storage.update_analysis_result(video_id, analysis_result)
                results[video_id] = analysis_result
                
                print(f"✅ Đã phân tích: {video_id}")
                
            except Exception as e:
                print(f"❌ Lỗi phân tích {video_id}: {e}")
                results[video_id] = f"Cảnh video cho sự kiện (ID: {video_id[:8]})"
        
        return results
    
    def generate_script_from_storage(self, video_ids: List[str], user_orders: Dict[str, int], 
                                   main_prompt: str, tone: str = "chuyên nghiệp") -> str:
        """
        Tạo kịch bản từ các video đã lưu trữ
        
        Args:
            video_ids (List[str]): Danh sách video IDs
            user_orders (Dict[str, int]): Thứ tự do user chỉ định
            main_prompt (str): Prompt chính
            tone (str): Tông giọng
            
        Returns:
            str: Kịch bản hoàn chỉnh
        """
        # Lấy thông tin videos
        video_infos = []
        for video_id in video_ids:
            info = self.storage.get_video_info(video_id)
            if info:
                video_infos.append(info)
        
        if not video_infos:
            return "❌ Không tìm thấy video nào để bắt đầu kịch bản"
        
        # Phân tích videos nếu chưa có
        for video_info in video_infos:
            if not video_info.analysis_result:
                print(f"🔄 Phân tích video chưa có kết quả: {video_info.original_name}")
                video_path = self.storage.get_video_for_analysis(video_info.id)
                if video_path:
                    analysis = self.analyzer.analyze_video_robust(video_path)
                    self.storage.update_analysis_result(video_info.id, analysis)
                    video_info.analysis_result = analysis
        
        # Tạo kịch bản
        script_content = self._generate_complete_script(video_infos, user_orders, main_prompt, tone)
        
        return script_content
    
    def _generate_complete_script(self, video_infos: List[VideoInfo], user_orders: Dict[str, int],
                                  main_prompt: str, tone: str) -> str:
        """Tạo kịch bản hoàn chỉnh, với voiceover liên kết."""
        try:
            ordered_videos = self._sort_videos_by_order(video_infos, user_orders, main_prompt)
            
            script_parts = []
            previous_voiceover = ""  # Để liên kết cảnh sau với trước
            
            for i, video_info in enumerate(ordered_videos):
                position = "opening" if i == 0 else ("ending" if i == len(ordered_videos)-1 else "middle")
                next_analysis = ordered_videos[i+1].analysis_result if i < len(ordered_videos)-1 else ""
                
                voiceover = self._generate_voiceover_for_scene(
                    video_info, main_prompt, tone, position, previous_voiceover, next_analysis
                )
                
                script_parts.append({
                    "scene_number": i + 1,
                    "video_file": video_info.original_name,
                    "duration": video_info.duration,
                    "description": self._extract_background_description(video_info.analysis_result or "Nội dung video chưa được phân tích"),
                    "voiceover": voiceover
                })
                
                previous_voiceover = voiceover  # Cập nhật cho cảnh sau
            
            total_duration = sum(v.duration for v in ordered_videos)
            return self._format_final_script(script_parts, main_prompt, total_duration)
            
        except Exception as e:
            print(f"❌ Lỗi tạo kịch bản: {e}")
            return f"❌ Không thể tạo kịch bản: {e}"
    
    def _extract_background_description(self, analysis_result: str) -> str:
        """Trích xuất chỉ phần 'Bối cảnh, môi trường chính' từ analysis_result sử dụng regex."""
        try:
            # Sử dụng regex để tìm phần 1. Bối cảnh, môi trường chính: và lấy đến trước phần 2.
            pattern = r"1\.\s*\*\*?Bối cảnh, môi trường chính:\*\*?\s*(.*?)(?=\n\s*2\.|$)"
            match = re.search(pattern, analysis_result, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            else:
                # Nếu không tìm thấy, trả về mô tả mặc định hoặc hết chuỗi
                return analysis_result.strip()
        except Exception as e:
            print(f"Lỗi trích xuất mô tả: {e}")
            return analysis_result.strip()
    
    def _sort_videos_by_order(self, video_infos: List[VideoInfo], user_orders: Dict[str, int], 
                              main_prompt: str) -> List[VideoInfo]:
        """Sắp xếp video theo thứ tự, tự đề xuất nếu không có user_orders"""
        # Nếu có user_orders, sắp xếp theo đó
        if user_orders:
            ordered_videos = []
            unordered_videos = []
            
            for video_info in video_infos:
                if video_info.original_name in user_orders:
                    ordered_videos.append((user_orders[video_info.original_name], video_info))
                else:
                    unordered_videos.append(video_info)
            
            # Sắp xếp videos có thứ tự
            ordered_videos.sort(key=lambda x: x[0])
            ordered_video_infos = [v[1] for v in ordered_videos]
            
            # Thêm videos chưa có thứ tự vào cuối
            return ordered_video_infos + unordered_videos
        
        # Nếu không có user_orders, dùng Gemini để đề xuất thứ tự
        else:
            print("🔄 Không có thứ tự người dùng, đang đề xuất thứ tự tự động bằng Gemini...")
            return self._suggest_order_with_gemini(video_infos, main_prompt)
    
    def _suggest_order_with_gemini(self, video_infos: List[VideoInfo], main_prompt: str) -> List[VideoInfo]:
        """Dùng Gemini để đề xuất thứ tự video dựa trên analysis và main_prompt"""
        try:
            # Chuẩn bị danh sách mô tả
            descriptions = []
            for info in video_infos:
                desc = f"{info.original_name}: {info.analysis_result or 'Chưa phân tích'}"
                descriptions.append(desc)
            
            descriptions_str = "\n".join(descriptions)
            
            prompt = f"""
            Dựa trên chủ đề kịch bản: "{main_prompt}"
            Và mô tả các cảnh video sau:
            {descriptions_str}
            
            Đề xuất thứ tự logic cho các cảnh để tạo kịch bản mạch lạc (ví dụ: bắt đầu từ giới thiệu, phát triển, kết thúc).
            Trả về dưới dạng danh sách thứ tự: 1: tên_file1.mp4, 2: tên_file2.mp4, ...
            Chỉ trả về danh sách này, không thêm giải thích.
            """
            
            response = self.analyzer.model.generate_content(prompt)
            if not response or not response.text:
                raise Exception("Gemini không trả về gợi ý thứ tự")
            
            # Parse kết quả (giả sử format như "1: file1.mp4\n2: file2.mp4")
            suggested_order = {}
            for line in response.text.strip().split("\n"):
                if ":" in line:
                    order_str, filename = line.split(":", 1)
                    try:
                        order = int(order_str.strip())
                        suggested_order[filename.strip()] = order
                    except ValueError:
                        continue
            
            # Sắp xếp dựa trên suggested_order
            ordered_videos = []
            unordered_videos = []
            
            for video_info in video_infos:
                if video_info.original_name in suggested_order:
                    ordered_videos.append((suggested_order[video_info.original_name], video_info))
                else:
                    unordered_videos.append(video_info)
            
            ordered_videos.sort(key=lambda x: x[0])
            ordered_video_infos = [v[1] for v in ordered_videos]
            
            print("✅ Đã đề xuất thứ tự tự động")
            return ordered_video_infos + unordered_videos
            
        except Exception as e:
            print(f"⚠️ Lỗi đề xuất thứ tự: {e}. Sử dụng thứ tự mặc định.")
            return video_infos  # Fallback về thứ tự gốc
    
    def _generate_voiceover_for_scene(self, video_info: VideoInfo, main_prompt: str, 
                                      tone: str, position: str, previous_voiceover: str = "", next_analysis: str = "") -> str:
        """Tạo voiceover dùng Claude cho chất lượng cao hơn, với liên kết cảnh."""
        # Comment hoặc xóa dòng tính max_words
        # max_words = int(video_info.duration * 150 / 60)  # ~150 từ/phút
        
        # Prompt chi tiết để Claude tạo voiceover hay, liên kết tốt, chung chung cho mọi chủ đề
        prompt = f"""
        Bạn là một biên kịch chuyên nghiệp, tạo voiceover script cho video dựa trên prompt người dùng. Đảm bảo voiceover:
        - Hấp dẫn, tự nhiên như kể chuyện, sử dụng ngôn ngữ phù hợp với chủ đề.
        - Nội dung voiceover bạn tạo ra phải bám sát với nội dung hình ảnh trên video, không được làm thông tin voiceover script sai lệch sự thật
        - Liên kết mượt mà với cảnh trước (tiếp nối ý từ voiceover trước) và dẫn dắt sang cảnh sau (dựa trên mô tả cảnh sau).
        - Tone: {tone}
        - Vị trí cảnh: {position} (opening: giới thiệu, middle: phát triển, ending: kết thúc).
        - Chủ đề tổng: {main_prompt}
        - Mô tả cảnh hiện tại: {video_info.analysis_result}
        - Voiceover cảnh trước (nếu có, tiếp nối ý): {previous_voiceover}
        - Mô tả cảnh sau (nếu có, dẫn dắt sang): {next_analysis}
        
        Chỉ trả về nội dung voiceover thuần túy, không thêm giải thích.
        """
            
        try:
            if self.use_openrouter:
                # Gọi qua OpenRouter với cấu hình tối ưu
                completion = self.claude_client.chat.completions.create(
                    model="anthropic/claude-3.5-sonnet",  # Cập nhật model đúng
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.7
                )
                response = completion.choices[0].message.content.strip()
            else:
                # Gọi trực tiếp Anthropic với cấu hình tối ưu
                message = self.claude_client.messages.create(
                    model="anthropic/claude-3.5-sonnet",  # Cập nhật model đúng
                    max_tokens=800,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                response = message.content[0].text.strip()
            
            if not response:
                raise Exception("Claude trả về kết quả rỗng")
                
            return response
            
        except Exception as e:
            error_details = f"Lỗi tạo voiceover với Claude: {str(e)}"
            print(error_details)
            
            # Log lỗi để debug
            with open("voiceover_errors.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now()}] Error Details: {error_details}")
                f.write(f"\n[{datetime.now()}] Prompt: {prompt}")
                f.write(f"\n[{datetime.now()}] Video Info: {video_info.original_name}\n")
            
            # Thử lại với prompt đơn giản hơn
            try:
                simplified_prompt = f"""
                Tạo voiceover ngắn gọn cho video này. Tone: {tone}. 
                Nội dung video: {video_info.analysis_result}
                """
                
                if self.use_openrouter:
                    completion = self.claude_client.chat.completions.create(
                        model="anthropic/claude-3.5-sonnet",  # Cập nhật model đúng
                        messages=[{"role": "user", "content": simplified_prompt}],
                        max_tokens=800,
                        temperature=0.7
                    )
                    return completion.choices[0].message.content.strip()
                else:
                    message = self.claude_client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=800,
                        temperature=0.7,
                        messages=[{"role": "user", "content": simplified_prompt}]
                    )
                    return message.content[0].text.strip()
                    
            except Exception as retry_error:
                print(f"Lỗi thử lại với Claude: {str(retry_error)}")
                # Log lỗi thử lại
                with open("voiceover_errors.log", "a", encoding="utf-8") as f:
                    f.write(f"\n[{datetime.now()}] Retry Error: {str(retry_error)}\n")
                
                return f"[Voiceover cho {video_info.original_name}] (Lỗi: Không thể kết nối với Claude. Vui lòng thử lại sau.)"
            
            # Thử backup với model khác
            try:
                # Fallback to GPT model
                backup_prompt = f"Tạo voiceover ngắn gọn cho video dựa trên: {video_info.analysis_result}"
                backup_completion = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": backup_prompt}],
                    max_tokens=200,
                    temperature=0.7
                )
                return backup_completion.choices[0].message.content.strip()
            except Exception as backup_error:
                print(f"Lỗi backup model: {str(backup_error)}")
                # Lưu lỗi để debug
                with open("voiceover_errors.log", "a") as f:
                    f.write(f"\n[{datetime.now()}] Primary Error: {error_details}")
                    f.write(f"\n[{datetime.now()}] Backup Error: {str(backup_error)}\n")
                return f"[Voiceover cho {video_info.original_name}] (Lỗi: Không thể kết nối với Claude. Vui lòng thử lại sau.)"
    
    def _edit_script(self, current_script: str, edit_prompt: str, main_prompt: str, tone: str) -> str:
        """Chỉnh sửa kịch bản dựa trên yêu cầu người dùng dùng Claude."""
        try:
            prompt = f"""
            Bạn là biên kịch chuyên nghiệp. Dựa trên kịch bản hiện tại sau:
            
            {current_script}
            
            Và yêu cầu chỉnh sửa từ người dùng: "{edit_prompt}"
            
            Hãy tạo kịch bản mới đã chỉnh sửa, giữ nguyên format chính xác (tiêu đề, Cảnh X, Video, Thời lượng, Hình ảnh, Voiceover).
            Giữ chủ đề tổng: {main_prompt}
            Tone: {tone}
            
            Chỉ trả về kịch bản mới hoàn chỉnh, không thêm giải thích.
            """
            
            message = self.claude_client.messages.create(
                model="claude-3",
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
            
        except Exception as e:
            print(f"Lỗi chỉnh sửa kịch bản: {e}")
            return current_script  # Fallback giữ nguyên
    

    
    def _format_final_script(self, script_parts: List[Dict], main_prompt: str, total_duration: float) -> str:
        """Format kịch bản cuối cùng"""
        # Làm tiêu đề động dựa trên main_prompt
        title = main_prompt.upper() if main_prompt else "KỊCH BẢN VIDEO"
        output = f"{title} (CÓ THỜI LƯỢNG VÀ VOICEOVER ĐẦY ĐỦ)\n\n"
        
        for part in script_parts:
            # Lọc chỉ lấy thông tin bối cảnh và hoạt động chính
            lines = part['description'].split('\n')
            context_line = next((line for line in lines if line.startswith('1.')), '')
            action_line = next((line for line in lines if line.startswith('3.')), '')
            filtered_description = '\n'.join(line for line in [context_line, action_line] if line)
            
            output += f"""Cảnh {part['scene_number']}
(Video: {part['video_file']})
Thời lượng: {part['duration']:.0f} giây
Hình ảnh:
{filtered_description}

Voiceover:
"{part['voiceover']}"

"""
        
        output += "(Kết thúc video với logo hoặc thông tin liên hệ)\n\n---KẾT THÚC KỊCH BẢN---"
        return output
    
    def cleanup(self):
        """Dọn dẹp resources"""
        self.analyzer.cleanup()

# Demo usage
def demo_storage_system():
    """Demo hệ thống lưu trữ"""
    print("🏪 DEMO HỆ THỐNG LỮU TRỮ VIDEO")
    print("="*40)
    
    api_key = input("Nhập Gemini API key: ")
    claude_api_key = input("Nhập Claude/OpenRouter API key: ")
    use_openrouter = input("Sử dụng OpenRouter? (y/n): ").strip().lower() == 'y'
    
    # Khởi tạo hệ thống
    generator = EnhancedVideoScriptGenerator(api_key, claude_api_key, use_openrouter, "./demo_storage")
    
    # Hiển thị stats
    stats = generator.storage.get_storage_stats()
    print(f"\n📊 Thống kê lưu trữ:")
    print(f"   Tổng videos: {stats['total_videos']}")
    print(f"   Dung lượng: {stats['total_size_mb']:.1f}MB")
    print(f"   Đã phân tích: {stats['analyzed_videos']}")
    
    # Lấy danh sách videos có sẵn
    existing_videos = generator.storage.get_all_videos()
    if existing_videos:
        print(f"\n📹 Videos đã lưu trữ:")
        for video in existing_videos[:5]:  # Hiển thị 5 video gần nhất
            print(f"   • {video.original_name} ({video.duration:.1f}s) - {video.status}")
    
    # Upload videos mới
    upload_choice = input("\nBạn có muốn upload videos mới? (y/n): ").strip().lower()
    if upload_choice == 'y':
        video_folder = input("Đường dẫn thư mục chứa videos: ")
        
        if os.path.exists(video_folder):
            video_files = []
            for file in os.listdir(video_folder):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(video_folder, file))
            
            if video_files:
                print(f"Tìm thấy {len(video_files)} videos")
                video_ids = generator.process_video_uploads(video_files)
                
                # Tạo kịch bản demo - Người dùng có thể thay main_prompt
                main_prompt = input("Nhập prompt chủ đề kịch bản (ví dụ: Tạo video giới thiệu sự kiện): ")
                tone = input("Nhập mô tả tông voiceover (ví dụ: Nhiệt huyết, truyền cảm hứng, nghiêm túc, chuyên nghiệp, hướng tới đối tượng trẻ em): ") or "chuyên nghiệp"
                # Giả sử không có user_orders để test tự động sắp xếp
                script = generator.generate_script_from_storage(video_ids, {}, main_prompt, tone)
                
                print("\n📝 KỊCH BẢN ĐÃ TẠO:")
                print("="*40)
                print(script)
                
                # Chức năng chỉnh sửa
                while True:
                    edit_choice = input("\nBạn có muốn chỉnh sửa kịch bản? (y/n): ").strip().lower()
                    if edit_choice != 'y':
                        break
                    
                    edit_prompt = input("Nhập yêu cầu chỉnh sửa (ví dụ: Làm voiceover cảnh 1 dài hơn): ")
                    script = generator._edit_script(script, edit_prompt, main_prompt, tone)
                    
                    print("\n📝 KỊCH BẢN SAU CHỈNH SỬA:")
                    print("="*40)
                    print(script)
    
    generator.cleanup()

if __name__ == "__main__":
    demo_storage_system()