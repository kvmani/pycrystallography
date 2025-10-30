import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

import { uploadCif } from '../services/api';
import { useDiffractionStore } from '../hooks/useDiffractionStore';

const dropzoneStyles: React.CSSProperties = {
  border: '2px dashed rgba(255, 255, 255, 0.2)',
  borderRadius: '16px',
  padding: '24px',
  textAlign: 'center',
  background: 'rgba(17, 25, 40, 0.65)',
  cursor: 'pointer',
  transition: 'border-color 0.2s ease-in-out, background 0.2s ease-in-out'
};

export function CifUpload() {
  const setStructure = useDiffractionStore((state) => state.setStructure);
  const setSummary = useDiffractionStore((state) => state.setSummary);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (!acceptedFiles.length) {
        return;
      }
      setError(null);
      setLoading(true);
      try {
        const file = acceptedFiles[0];
        const response = await uploadCif(file);
        setStructure(response.structure, response.summary);
        setSummary(response.summary);
      } catch (err) {
        console.error(err);
        setError('Failed to parse CIF file. Please ensure the file is valid.');
      } finally {
        setLoading(false);
      }
    },
    [setStructure, setSummary]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': ['.cif'],
      'application/octet-stream': ['.cif']
    }
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <div
        {...getRootProps()}
        style={{
          ...dropzoneStyles,
          borderColor: isDragActive ? '#4f46e5' : dropzoneStyles.border,
          background: isDragActive ? 'rgba(79, 70, 229, 0.2)' : dropzoneStyles.background
        }}
      >
        <input {...getInputProps()} />
        {loading ? (
          <p>Parsing CIF…</p>
        ) : (
          <p>{isDragActive ? 'Drop the CIF file here' : 'Drag & drop a CIF file here, or click to browse'}</p>
        )}
      </div>
      {error ? <span style={{ color: '#f97316' }}>{error}</span> : null}
    </div>
  );
}
