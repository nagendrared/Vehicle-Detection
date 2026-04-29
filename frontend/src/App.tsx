import React, { useState, useCallback, useRef, useEffect } from 'react';
import { 
  Upload, Loader2, Trash2, AlertCircle, ImageIcon, Car,
  Sun, Moon, Download, List, ZoomIn, ZoomOut,
  Share2, Info, Settings, Filter, Sliders, Save, RefreshCw, Maximize, Minimize
} from 'lucide-react';

// ─── Constants ────────────────────────────────────────────────────────────────
const MAX_FILE_SIZE = 5 * 1024 * 1024;
// Must match backend ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const;
const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'] as const;

// Must match backend default CONF_THRESH = 0.25
const DEFAULT_MIN_CONFIDENCE = 0.25;
const DEFAULT_MAX_RESULTS = 10;

// Backend base URL – single source of truth
const API_BASE = 'http://127.0.0.1:5000';

// ─── Types ────────────────────────────────────────────────────────────────────
type FileValidationResult = {
  isValid: boolean;
  error?: string;
};

type DetectedObject = {
  label: string;
  score: number;
  box: number[];
  dimensions?: {
    width: number;
    height: number;
    area: number;
  };
};

// Shape returned by the backend /detect endpoint
type DetectResponse = {
  success: boolean;
  count: number;
  detections: DetectedObject[];
  /** Base-64 encoded JPEG of the annotated image */
  image_base64: string;
  processing_time: number;
  error?: string;
};

type FilterSettings = {
  minConfidence: number;
  maxResults: number;
  selectedClasses: string[];
};

// ─── App ──────────────────────────────────────────────────────────────────────
function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [darkMode, setDarkMode] = useState(false);
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([]);
  const [showDetectedList, setShowDetectedList] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [processingTime, setProcessingTime] = useState<number | null>(null);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [filterSettings, setFilterSettings] = useState<FilterSettings>({
    // Default matches backend CONF_THRESH = 0.25
    minConfidence: DEFAULT_MIN_CONFIDENCE,
    maxResults: DEFAULT_MAX_RESULTS,
    selectedClasses: []
  });
  const [availableClasses, setAvailableClasses] = useState<string[]>([]);
  const [processingHistory, setProcessingHistory] = useState<{
    filename: string;
    timestamp: Date;
    objectCount: number;
    thumbnail?: string;
  }[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonImage, setComparisonImage] = useState<string | null>(null);
  const [batchProcessing, setBatchProcessing] = useState(false);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchResults, setBatchResults] = useState<{
    filename: string;
    detections: DetectedObject[];
    imageUrl: string;
  }[]>([]);
  const fullscreenRef = useRef<HTMLDivElement>(null);

  // ─── Helpers ──────────────────────────────────────────────────────────────
  const validateFile = useCallback((file: File): FileValidationResult => {
    if (!file) return { isValid: false, error: 'No file selected' };

    if (!ALLOWED_TYPES.includes(file.type as typeof ALLOWED_TYPES[number])) {
      return { 
        isValid: false, 
        error: `Invalid file type. Please upload ${ALLOWED_EXTENSIONS.join(', ')} files only` 
      };
    }

    if (file.size > MAX_FILE_SIZE) {
      return { 
        isValid: false, 
        error: `File size must be less than ${formatFileSize(MAX_FILE_SIZE)}` 
      };
    }

    return { isValid: true };
  }, []);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  /**
   * Build a FormData payload that exactly matches what the backend expects:
   *   - field "file"            (backend: request.files["file"])
   *   - field "min_confidence"  (backend: request.form.get("min_confidence"))
   *   - field "max_results"     (backend: request.form.get("max_results"))
   *   - field "classes"         (backend: request.form.get("classes"))  comma-separated string
   */
  const buildFormData = (file: File, settings: FilterSettings): FormData => {
    const formData = new FormData();

    // KEY FIX: backend expects "file", not "image"
    formData.append('file', file);

    formData.append('min_confidence', settings.minConfidence.toString());
    formData.append('max_results', settings.maxResults.toString());

    if (settings.selectedClasses.length > 0) {
      formData.append('classes', settings.selectedClasses.join(','));
    }

    return formData;
  };

  /**
   * Parse the backend DetectResponse and return normalised detections.
   * KEY FIX: result image comes from response.image_base64, not a separate URL.
   */
  const parseDetectResponse = (data: DetectResponse) => {
    if (!data.success) {
      throw new Error(data.error || 'Detection failed on the server');
    }

    const processedObjects: DetectedObject[] = (data.detections ?? []).map((obj: any) => ({
      label: obj.label || 'Unknown',
      score: typeof obj.score === 'number' ? obj.score : 0,
      box: Array.isArray(obj.box) ? obj.box : [],
      dimensions: obj.dimensions ?? null
    }));

    // Backend returns a base64-encoded JPEG – convert to a data URL
    const annotatedImageUrl = `data:image/jpeg;base64,${data.image_base64}`;

    return {
      processedObjects,
      annotatedImageUrl,
      processingTime: data.processing_time ?? null
    };
  };

  // ─── Single-image submission ───────────────────────────────────────────────
  const handleSubmit = useCallback(async (event?: React.FormEvent) => {
    event?.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);
    setUploadProgress(0);
    setProcessingTime(null);

    try {
      const formData = buildFormData(selectedFile, filterSettings);

      const response = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Detection failed: ${response.statusText}`);
      }

      const data: DetectResponse = await response.json();
      const { processedObjects, annotatedImageUrl, processingTime: pt } = parseDetectResponse(data);

      // KEY FIX: use base64 data URL, not a non-existent server endpoint
      setResultImage(annotatedImageUrl);
      setProcessingTime(pt);

      if (processedObjects.length > 0) {
        setDetectedObjects(processedObjects);
        setShowDetectedList(true);

        const classes = [...new Set(processedObjects.map(obj => obj.label))] as string[];
        setAvailableClasses(prev => [...new Set([...prev, ...classes])]);

        // Thumbnail from the local preview (no extra network round-trip)
        const thumbnailCanvas = document.createElement('canvas');
        thumbnailCanvas.width = 100;
        thumbnailCanvas.height = 100;
        const img = new Image();
        img.onload = () => {
          const ctx = thumbnailCanvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(img, 0, 0, 100, 100);
            setProcessingHistory(prev => [
              {
                filename: selectedFile.name,
                timestamp: new Date(),
                objectCount: processedObjects.length,
                thumbnail: thumbnailCanvas.toDataURL('image/jpeg')
              },
              ...prev.slice(0, 9)
            ]);
          }
        };
        img.src = previewUrl || '';
      } else {
        setDetectedObjects([]);
        setError('No vehicles detected in the image');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image');
      console.error(err);
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  }, [selectedFile, filterSettings, previewUrl]);

  // ─── Download ──────────────────────────────────────────────────────────────
  const handleDownload = useCallback(() => {
    if (resultImage) {
      const link = document.createElement('a');
      link.href = resultImage;
      link.download = `detected_${selectedFile?.name || 'image.jpg'}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }, [resultImage, selectedFile]);

  // ─── Zoom ─────────────────────────────────────────────────────────────────
  const handleZoomIn  = useCallback(() => setZoomLevel(prev => Math.min(prev + 0.1, 2)), []);
  const handleZoomOut = useCallback(() => setZoomLevel(prev => Math.max(prev - 0.1, 0.5)), []);

  // ─── Clear ────────────────────────────────────────────────────────────────
  const clearImage = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResultImage(null);
    setError(null);
    setDetectedObjects([]);
    setShowDetectedList(false);
    setZoomLevel(1);
    setComparisonImage(null);
    setIsComparing(false);
    setProcessingTime(null);
  }, []);

  // ─── Drag-and-drop ────────────────────────────────────────────────────────
  const handleDrop = useCallback((e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const validation = validateFile(file);
    if (!validation.isValid) { setError(validation.error); return; }
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onloadstart = () => setIsPreviewLoading(true);
    reader.onload = () => { setPreviewUrl(reader.result as string); setIsPreviewLoading(false); };
    reader.readAsDataURL(file);
  }, [validateFile]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent<HTMLLabelElement>) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.currentTarget.click(); }
  }, []);

  // ─── Fullscreen ───────────────────────────────────────────────────────────
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      fullscreenRef.current?.requestFullscreen().catch(err => {
        setError(`Error enabling fullscreen: ${err.message}`);
      });
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  // ─── Batch processing ─────────────────────────────────────────────────────
  const handleBatchUpload = useCallback((files: FileList) => {
    const validFiles: File[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (validateFile(file).isValid) validFiles.push(file);
    }
    if (validFiles.length > 0) {
      setBatchFiles(validFiles);
      setBatchProcessing(true);
    } else {
      setError('No valid files found for batch processing');
    }
  }, [validateFile]);

  const processBatchFiles = useCallback(async () => {
    if (batchFiles.length === 0) return;

    setBatchResults([]);
    setBatchProgress(0);
    setIsLoading(true);

    const results: { filename: string; detections: DetectedObject[]; imageUrl: string }[] = [];

    for (let i = 0; i < batchFiles.length; i++) {
      const file = batchFiles[i];

      try {
        // KEY FIX: use "file" field and API_BASE constant
        const formData = buildFormData(file, filterSettings);

        const response = await fetch(`${API_BASE}/detect`, { method: 'POST', body: formData });

        if (!response.ok) throw new Error(`Detection failed for ${file.name}: ${response.statusText}`);

        const data: DetectResponse = await response.json();
        const { processedObjects, annotatedImageUrl } = parseDetectResponse(data);

        // KEY FIX: imageUrl is the base64 data URL from the response, not a server file path
        results.push({ filename: file.name, detections: processedObjects, imageUrl: annotatedImageUrl });

        const classes = [...new Set(processedObjects.map(obj => obj.label))] as string[];
        setAvailableClasses(prev => [...new Set([...prev, ...classes])]);
      } catch (err) {
        console.error(`Error processing ${file.name}:`, err);
      }

      setBatchProgress(Math.round(((i + 1) / batchFiles.length) * 100));
    }

    setBatchResults(results);
    setIsLoading(false);
  }, [batchFiles, filterSettings]);

  // ─── History ──────────────────────────────────────────────────────────────
  const selectFromHistory = useCallback((index: number) => {
    const historyItem = processingHistory[index];
    if (!historyItem) return;
    setError(`Loading previous detection: ${historyItem.filename}`);
    setShowHistory(false);
  }, [processingHistory]);

  // ─── Comparison mode ──────────────────────────────────────────────────────
  const toggleComparisonMode = useCallback(() => {
    setIsComparing(prev => {
      if (!prev && resultImage) setComparisonImage(resultImage);
      else setComparisonImage(null);
      return !prev;
    });
  }, [resultImage]);

  // ─── Effects ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (batchProcessing && batchFiles.length > 0) processBatchFiles();
  }, [batchProcessing, batchFiles, processBatchFiles]);

  useEffect(() => {
    const handleFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div 
      ref={fullscreenRef}
      className={`min-h-screen transition-colors duration-300 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gradient-to-b from-blue-50 to-white'
      }`}
    >
      {/* ── Header ── */}
      <div className={`${
        darkMode ? 'bg-gray-800' : 'bg-gradient-to-r from-blue-600 to-blue-700'
      } py-6 sm:py-8 px-4 sm:px-6 shadow-lg mb-6 sm:mb-10 transition-colors duration-300`}>
        <div className="max-w-7xl mx-auto relative">
          <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 rounded-full hover:bg-white/20 transition-all duration-300"
              aria-label="Settings"
            >
              <Settings className="w-6 h-6 text-white" />
            </button>
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="p-2 rounded-full hover:bg-white/20 transition-all duration-300"
              aria-label="History"
            >
              <RefreshCw className="w-6 h-6 text-white" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="p-2 rounded-full hover:bg-white/20 transition-all duration-300"
              aria-label="Toggle fullscreen"
            >
              {isFullscreen ? <Minimize className="w-6 h-6 text-white" /> : <Maximize className="w-6 h-6 text-white" />}
            </button>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2 rounded-full hover:bg-white/20 transition-all duration-300"
              aria-label="Toggle dark mode"
            >
              {darkMode ? <Sun className="w-6 h-6 text-yellow-300" /> : <Moon className="w-6 h-6 text-white" />}
            </button>
          </div>
          <div className="flex items-center justify-center gap-3 text-white">
            <Car className="w-8 h-8 sm:w-10 sm:h-10" />
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">Vehicle Detection</h1>
          </div>
          <p className={`${darkMode ? 'text-gray-300' : 'text-blue-100'} text-center mt-2 text-sm sm:text-base`}>
            Upload an image to detect and analyze vehicles
          </p>
        </div>
      </div>

      {/* ── Settings Panel ── */}
      {showSettings && (
        <div className={`container mx-auto px-4 sm:px-6 mb-6 ${
          darkMode ? 'bg-gray-800' : 'bg-white'
        } p-6 rounded-xl shadow-lg max-w-3xl`}>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <Sliders className="w-5 h-5 text-blue-500" />
              Detection Settings
            </h2>
            <button onClick={() => setShowSettings(false)} className="text-gray-500 hover:text-gray-700">✕</button>
          </div>

          <div className="space-y-4">
            {/* Confidence slider – range matches backend: 0.0–1.0 */}
            <div>
              <label className="block text-sm font-medium mb-1">
                Minimum Confidence ({Math.round(filterSettings.minConfidence * 100)}%)
              </label>
              <input
                type="range" min="0.05" max="0.95" step="0.05"
                value={filterSettings.minConfidence}
                onChange={(e) => setFilterSettings(prev => ({ ...prev, minConfidence: parseFloat(e.target.value) }))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>5%</span><span>95%</span>
              </div>
            </div>

            {/* Max results slider – matches backend max_results param */}
            <div>
              <label className="block text-sm font-medium mb-1">
                Maximum Results ({filterSettings.maxResults})
              </label>
              <input
                type="range" min="1" max="50" step="1"
                value={filterSettings.maxResults}
                onChange={(e) => setFilterSettings(prev => ({ ...prev, maxResults: parseInt(e.target.value) }))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>1</span><span>50</span>
              </div>
            </div>

            {/* Class filter – sent as comma-separated string matching backend classes param */}
            {availableClasses.length > 0 && (
              <div>
                <label className="block text-sm font-medium mb-2">Filter by Classes</label>
                <div className="flex flex-wrap gap-2">
                  {availableClasses.map(className => (
                    <button
                      key={className}
                      onClick={() => setFilterSettings(prev => {
                        const isSelected = prev.selectedClasses.includes(className);
                        return {
                          ...prev,
                          selectedClasses: isSelected
                            ? prev.selectedClasses.filter(c => c !== className)
                            : [...prev.selectedClasses, className]
                        };
                      })}
                      className={`px-3 py-1 text-sm rounded-full ${
                        filterSettings.selectedClasses.includes(className)
                          ? 'bg-blue-500 text-white'
                          : darkMode ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                      } transition-colors`}
                    >
                      {className}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="flex justify-end pt-2 gap-2">
              <button
                onClick={() => setFilterSettings({ minConfidence: DEFAULT_MIN_CONFIDENCE, maxResults: DEFAULT_MAX_RESULTS, selectedClasses: [] })}
                className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'} transition-colors`}
              >
                Reset
              </button>
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── History Panel ── */}
      {showHistory && processingHistory.length > 0 && (
        <div className={`container mx-auto px-4 sm:px-6 mb-6 ${
          darkMode ? 'bg-gray-800' : 'bg-white'
        } p-6 rounded-xl shadow-lg max-w-3xl`}>
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <RefreshCw className="w-5 h-5 text-blue-500" />
              Recent Detections
            </h2>
            <button onClick={() => setShowHistory(false)} className="text-gray-500 hover:text-gray-700">✕</button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
            {processingHistory.map((item, index) => (
              <div
                key={index}
                onClick={() => selectFromHistory(index)}
                className={`cursor-pointer rounded-lg overflow-hidden border ${
                  darkMode ? 'border-gray-700 hover:border-gray-500' : 'border-gray-200 hover:border-gray-400'
                } transition-colors`}
              >
                {item.thumbnail && (
                  <div className="h-20 bg-gray-100 dark:bg-gray-700">
                    <img src={item.thumbnail} alt={item.filename} className="w-full h-full object-cover" />
                  </div>
                )}
                <div className="p-2">
                  <p className="text-xs truncate">{item.filename}</p>
                  <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
                    <span>{item.objectCount} vehicles</span>
                    <span>{item.timestamp.toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Main Content ── */}
      <div className="container mx-auto px-4 sm:px-6 flex flex-col items-center justify-center max-w-7xl">

        {/* Upload area – shown only when no image is selected */}
        {!previewUrl && !batchProcessing && (
          <div className={`${
            darkMode ? 'bg-gray-800' : 'bg-white'
          } p-6 sm:p-8 rounded-2xl shadow-xl flex flex-col items-center w-full max-w-xs sm:max-w-sm md:max-w-xl lg:max-w-3xl mb-6 sm:mb-8`}>
            {/* Batch trigger */}
            <div className="flex gap-4 w-full mb-6">
              <label className={`flex-1 px-4 py-3 rounded-xl cursor-pointer ${
                darkMode 
                  ? 'bg-purple-600 hover:bg-purple-700' 
                  : 'bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700'
              } text-white transition-all duration-300 flex items-center justify-center gap-2`}>
                <Filter className="w-5 h-5" />
                Batch Process
                <input
                  type="file" multiple accept={ALLOWED_TYPES.join(',')} className="hidden"
                  onChange={(e) => { if (e.target.files && e.target.files.length > 0) handleBatchUpload(e.target.files); }}
                />
              </label>
            </div>

            {/* Single-file drop zone */}
            <label
              className={`cursor-pointer flex flex-col items-center gap-4 sm:gap-6 p-8 sm:p-10 border-2 border-dashed rounded-xl w-full 
                ${isDragOver 
                  ? 'border-blue-500 bg-blue-50' 
                  : darkMode 
                    ? 'border-gray-600 hover:border-blue-500 hover:bg-gray-700' 
                    : 'border-gray-300 hover:border-blue-500 hover:bg-blue-50'
                }
                transition-all duration-300 focus-within:ring-4 focus-within:ring-blue-200`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              onKeyDown={handleKeyPress}
              tabIndex={0}
              role="button"
              aria-label="Upload image"
            >
              <Upload className={`w-12 h-12 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
              <div className="text-center">
                <p className={`text-lg font-medium ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                  Drag and drop your image here
                </p>
                <p className={`mt-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>or click to browse</p>
                <p className={`mt-2 text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  Supported formats: {ALLOWED_EXTENSIONS.join(', ')}
                </p>
                <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  Max file size: {formatFileSize(MAX_FILE_SIZE)}
                </p>
              </div>
              <input
                type="file" className="hidden" accept={ALLOWED_TYPES.join(',')}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  const validation = validateFile(file);
                  if (!validation.isValid) { setError(validation.error); return; }
                  setSelectedFile(file);
                  const reader = new FileReader();
                  reader.onloadstart = () => setIsPreviewLoading(true);
                  reader.onload = () => { setPreviewUrl(reader.result as string); setIsPreviewLoading(false); };
                  reader.readAsDataURL(file);
                }}
              />
            </label>
          </div>
        )}

        {/* ── Batch Processing UI ── */}
        {batchProcessing && (
          <div className={`${
            darkMode ? 'bg-gray-800' : 'bg-white'
          } p-6 rounded-2xl shadow-xl flex flex-col items-center w-full max-w-4xl mb-6`}>
            <div className="flex justify-between items-center w-full mb-4">
              <h2 className={`text-xl sm:text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} flex items-center gap-2`}>
                <Filter className="w-6 h-6 text-blue-500" />
                Batch Processing
              </h2>
              <button
                onClick={() => { setBatchProcessing(false); setBatchFiles([]); setBatchResults([]); }}
                className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}
                aria-label="Close batch processing"
              >
                ✕
              </button>
            </div>

            {isLoading ? (
              <div className="w-full">
                <div className="mb-2 flex justify-between">
                  <span>Processing {batchFiles.length} files…</span>
                  <span>{batchProgress}%</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                  <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${batchProgress}%` }} />
                </div>
              </div>
            ) : (
              <>
                {batchResults.length > 0 ? (
                  <div className="w-full">
                    <p className="mb-4">Processed {batchResults.length} files successfully.</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                      {batchResults.map((result, index) => (
                        <div key={index} className={`${darkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg overflow-hidden`}>
                          {/* KEY FIX: imageUrl is now a base64 data URL – renders correctly */}
                          <img src={result.imageUrl} alt={result.filename} className="w-full h-40 object-cover" />
                          <div className="p-3">
                            <p className="font-medium truncate">{result.filename}</p>
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              {result.detections.length} vehicles detected
                            </p>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {[...new Set(result.detections.map(d => d.label))].slice(0, 3).map(label => (
                                <span key={label} className="px-2 py-0.5 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full">
                                  {label}
                                </span>
                              ))}
                              {[...new Set(result.detections.map(d => d.label))].length > 3 && (
                                <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200 rounded-full">
                                  +{[...new Set(result.detections.map(d => d.label))].length - 3} more
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-6 flex justify-center">
                      <button
                        onClick={() => { setBatchProcessing(false); setBatchFiles([]); setBatchResults([]); }}
                        className={`px-6 py-2 ${
                          darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700'
                        } text-white rounded-lg transition-colors`}
                      >
                        Done
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <p>Ready to process {batchFiles.length} files.</p>
                    <button
                      onClick={processBatchFiles}
                      className={`mt-4 px-6 py-2 ${
                        darkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700'
                      } text-white rounded-lg transition-colors`}
                    >
                      Start Processing
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* ── Image Preview + Result ── */}
        {previewUrl && (
          <div className="w-full max-w-full lg:max-w-6xl px-2 sm:px-4">
            <div className={`grid ${resultImage ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'} gap-6 sm:gap-8`}>
              {/* Original image */}
              <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 rounded-2xl shadow-xl flex flex-col items-center transition-all duration-300`}>
                <div className="flex justify-between items-center w-full mb-4">
                  <h2 className={`text-xl sm:text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} flex items-center gap-2`}>
                    <ImageIcon className="w-6 h-6 text-blue-500" />
                    Uploaded Image
                  </h2>
                  {selectedFile && (
                    <span className={`text-sm ${darkMode ? 'text-gray-300 bg-gray-700' : 'text-gray-500 bg-gray-100'} px-3 py-1 rounded-full`}>
                      {formatFileSize(selectedFile.size)}
                    </span>
                  )}
                </div>
                <div className={`relative w-full ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-xl p-4`}>
                  <img
                    src={previewUrl}
                    alt="Uploaded"
                    className="w-full h-48 sm:h-64 md:h-80 lg:h-96 object-contain rounded-lg shadow-sm transition-transform duration-300 hover:scale-[1.02]"
                  />
                </div>
              </div>

              {/* Annotated result */}
              {resultImage && (
                <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} p-6 rounded-2xl shadow-xl flex flex-col items-center transition-all duration-300`}>
                  <div className="flex justify-between items-center w-full mb-4">
                    <h2 className={`text-xl sm:text-2xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'} flex items-center gap-2`}>
                      <Car className="w-6 h-6 text-blue-500" />
                      Detected Vehicles
                      {processingTime !== null && (
                        <span className={`ml-2 text-xs font-normal px-2 py-1 rounded-full ${
                          darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-500'
                        }`}>
                          {processingTime}s
                        </span>
                      )}
                    </h2>
                    <div className="flex items-center gap-2">
                      <button onClick={handleZoomOut} className="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors" aria-label="Zoom out">
                        <ZoomOut className="w-5 h-5" />
                      </button>
                      <button onClick={handleZoomIn} className="p-1.5 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors" aria-label="Zoom in">
                        <ZoomIn className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  <div className={`relative w-full ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-xl p-4`}>
                    {isComparing && comparisonImage ? (
                      <div className="grid grid-cols-2 gap-2">
                        <img src={previewUrl || ''} alt="Original" className="w-full object-contain rounded-lg shadow-sm" style={{ transform: `scale(${zoomLevel})` }} />
                        <img src={resultImage} alt="Detected objects" className="w-full object-contain rounded-lg shadow-sm" style={{ transform: `scale(${zoomLevel})` }} />
                      </div>
                    ) : (
                      <img
                        src={resultImage}
                        alt="Detected objects"
                        className="w-full h-48 sm:h-64 md:h-80 lg:h-96 object-contain rounded-lg shadow-sm transition-transform duration-300"
                        style={{ transform: `scale(${zoomLevel})` }}
                      />
                    )}
                  </div>

                  <div className="flex gap-4 mt-4 w-full">
                    <button onClick={handleDownload} className="flex-1 px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors duration-300 flex items-center justify-center gap-2">
                      <Download className="w-5 h-5" />
                      Download
                    </button>
                    <button onClick={() => setShowDetectedList(!showDetectedList)} className="flex-1 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors duration-300 flex items-center justify-center gap-2">
                      <List className="w-5 h-5" />
                      {showDetectedList ? 'Hide' : 'Show'} Vehicles
                    </button>
                    <button
                      onClick={toggleComparisonMode}
                      className={`flex-1 px-4 py-2 ${isComparing ? 'bg-purple-500 hover:bg-purple-600' : 'bg-gray-500 hover:bg-gray-600'} text-white rounded-lg transition-colors duration-300 flex items-center justify-center gap-2`}
                    >
                      <Share2 className="w-5 h-5" />
                      {isComparing ? 'Hide' : 'Compare'}
                    </button>
                  </div>

                  {/* Detection list */}
                  {showDetectedList && detectedObjects.length > 0 && (
                    <div className={`mt-4 w-full ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} rounded-lg p-4`}>
                      <h3 className="text-lg font-semibold mb-2">Detected Vehicles:</h3>
                      <ul className="space-y-2">
                        {detectedObjects.map((obj, index) => (
                          <li key={index} className={`flex justify-between items-center ${darkMode ? 'bg-gray-600' : 'bg-white'} p-3 rounded-lg shadow-sm`}>
                            <div className="flex flex-col">
                              <span className="font-medium">{obj.label}</span>
                              {obj.dimensions && (
                                <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                  Area: {Math.round(obj.dimensions.area)}px²
                                </span>
                              )}
                            </div>
                            <span className={`${
                              obj.score > 0.7 ? 'bg-green-100 text-green-800' :
                              obj.score > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                              'bg-red-100 text-red-800'
                            } px-3 py-1 rounded-full text-sm font-medium`}>
                              {Math.round(obj.score * 100)}%
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Action buttons ── */}
            <div className="flex flex-col sm:flex-row justify-center gap-4 sm:gap-6 mt-6 sm:mt-8">
              <button
                onClick={handleSubmit}
                className={`px-6 sm:px-8 py-3 rounded-xl text-base sm:text-lg font-medium
                  ${isLoading 
                    ? 'bg-blue-400 cursor-not-allowed' 
                    : darkMode
                      ? 'bg-blue-600 hover:bg-blue-700'
                      : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700'
                  } 
                  text-white transition-all duration-300 transform hover:scale-105 active:scale-95
                  shadow-lg hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-blue-200`}
                disabled={isLoading}
                aria-label={isLoading ? 'Processing image' : 'Detect vehicles'}
              >
                {isLoading ? (
                  <><Loader2 className="w-5 h-5 animate-spin inline-block mr-2" /><span>Processing…</span></>
                ) : (
                  'Detect Vehicles'
                )}
              </button>

              <button
                onClick={clearImage}
                className={`px-6 sm:px-8 py-3 rounded-xl text-base sm:text-lg font-medium
                  ${darkMode
                    ? 'bg-red-600 hover:bg-red-700'
                    : 'bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700'
                  }
                  text-white transition-all duration-300 transform hover:scale-105 active:scale-95
                  shadow-lg hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-red-200
                  flex items-center justify-center gap-2`}
                aria-label="Delete image"
              >
                <Trash2 className="w-5 h-5" />
                Delete Image
              </button>

              {resultImage && (
                <button
                  onClick={() => alert('Settings saved as preset')}
                  className={`px-6 sm:px-8 py-3 rounded-xl text-base sm:text-lg font-medium
                    ${darkMode
                      ? 'bg-green-600 hover:bg-green-700'
                      : 'bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700'
                    }
                    text-white transition-all duration-300 transform hover:scale-105 active:scale-95
                    shadow-lg hover:shadow-xl focus:outline-none focus:ring-4 focus:ring-green-200
                    flex items-center justify-center gap-2`}
                  aria-label="Save settings"
                >
                  <Save className="w-5 h-5" />
                  Save Settings
                </button>
              )}
            </div>
          </div>
        )}

        {/* ── Error banner ── */}
        {error && (
          <div className={`flex items-center gap-3 mt-6 ${
            darkMode ? 'bg-red-900 border-red-700' : 'bg-red-50 border-red-200'
          } border px-6 py-4 rounded-xl shadow-sm`}>
            <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0" />
            <p className={darkMode ? 'text-red-200' : 'text-red-600'}>{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;