import React from "react"
import PDFDiffStreamlitWrapper from "./PDFDiff"
import { createRoot } from 'react-dom/client';

const container = document.getElementById('root');
const root = createRoot(container!);
root.render(  <React.StrictMode>
    <PDFDiffStreamlitWrapper/>
</React.StrictMode>);