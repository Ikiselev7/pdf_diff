import {
    Streamlit,
    StreamlitComponentBase,
    withStreamlitConnection,
} from "streamlit-component-lib";
import React, { MouseEventHandler, ReactNode, createContext, memo, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf/dist/esm/entry.webpack";
import { Stage, Layer, Rect, Group, Text } from "react-konva";

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.js`;


enum EntityType {
    Insert = 'insert',
    Delete = 'delete',
    Replace = 'replace'
}


function getDloPageMap(dlos: Dlo[]): Map<string, number> {
    const map = new Map<string, number>();

    dlos.forEach(dlo => {
        map.set(dlo.id, dlo.page);
    });

    return map;
}


function getDloByPage(dlos: Dlo[]): Map<number, Dlo[]> {
    const map = new Map<number, Dlo[]>();

    dlos.forEach(dlo => {
        const pageDlos = map.get(dlo.page);

        if (pageDlos) {
            pageDlos.push(dlo);
        } else {
            map.set(dlo.page, [dlo]);
        }
    });

    return map;
}


function transformEntities(entities: any[]): [Dlo[], Dlo[]] {
    const sourceDlos: Dlo[] = [];
    const targetDlos: Dlo[] = [];

    entities.forEach(entity => {
        switch(entity.type) {
            case EntityType.Insert: 
                targetDlos.push({
                    id: entity.id,
                    box: {x1: entity.new_dlo.box[0], y1: entity.new_dlo.box[1], x2: entity.new_dlo.box[2], y2: entity.new_dlo.box[3]},
                    label: entity.new_dlo.label,
                    score: entity.new_dlo.score,
                    page: entity.new_page,
                    type: entity.type,
                    isSource: false,
                    diff: entity.diff.map((diff: any) => ({
                        id: diff.id,
                        box: diff.b_box_new.map((box: number[]) => ({x1: box[0], y1: box[1], x2: box[2], y2: box[3]})),
                        type: diff.type
                    }))
                });
                break;

            case EntityType.Delete: 
                sourceDlos.push({
                    id: entity.id,
                    box: {x1: entity.old_dlo.box[0], y1: entity.old_dlo.box[1], x2: entity.old_dlo.box[2], y2: entity.old_dlo.box[3]},
                    label: entity.old_dlo.label,
                    score: entity.old_dlo.score,
                    page: entity.old_page,
                    type: entity.type,
                    isSource: true,
                    diff: entity.diff.map((diff: any) => ({
                        id: diff.id,
                        box: diff.b_box_old.map((box: number[]) => ({x1: box[0], y1: box[1], x2: box[2], y2: box[3]})),
                        type: diff.type
                    }))
                });
                break;

            case EntityType.Replace:
                const oldDlo: Dlo = {
                    id: entity.id,
                    box: {x1: entity.old_dlo.box[0], y1: entity.old_dlo.box[1], x2: entity.old_dlo.box[2], y2: entity.old_dlo.box[3]},
                    label: entity.old_dlo.label,
                    score: entity.old_dlo.score,
                    page: entity.old_page,
                    type: entity.type,
                    isSource: true,
                    diff: []
                };

                const newDlo: Dlo = {
                    id: entity.id,
                    box: {x1: entity.new_dlo.box[0], y1: entity.new_dlo.box[1], x2: entity.new_dlo.box[2], y2: entity.new_dlo.box[3]},
                    label: entity.new_dlo.label,
                    score: entity.new_dlo.score,
                    page: entity.new_page,
                    type: entity.type,
                    isSource: false,
                    diff: []
                };

                entity.diff.forEach((diff: any) => {
                    const diffElement: DiffElement = {
                        id: diff.id,
                        box: [],
                        type: diff.type
                    };

                    if (diff.type === EntityType.Insert) {
                        diffElement.box = diff.b_box_new.map((box: number[]) => ({x1: box[0], y1: box[1], x2: box[2], y2: box[3]}));
                        newDlo.diff.push(diffElement);
                    } else if (diff.type === EntityType.Delete) {
                        diffElement.box = diff.b_box_old.map((box: number[]) => ({x1: box[0], y1: box[1], x2: box[2], y2: box[3]}));
                        oldDlo.diff.push(diffElement);
                    } else if (diff.type === EntityType.Replace) {
                        const oldDiffElement = {...diffElement, box: diff.b_box_old.map((box: number[]) => ({x1: box[0], y1: box[1], x2: box[2], y2: box[3]}))};
                        const newDiffElement = {...diffElement, box: diff.b_box_new.map((box: number[]) => ({x1: box[0], y1: box[1], x2: box[2], y2: box[3]}))};
                        oldDlo.diff.push(oldDiffElement);
                        newDlo.diff.push(newDiffElement);
                    }
                });

                sourceDlos.push(oldDlo);
                targetDlos.push(newDlo);
                break;

            default: 
                break;
        }
    });

    return [sourceDlos, targetDlos];
}


interface DrawRectsProps {
    diff: DiffElement;
    index: string;
    isSource: boolean;
}

const DrawRects: React.FC<DrawRectsProps> = React.memo(({ diff, index, isSource }) => {
    const { hoveredIds, setHoveredId } = useContext(DiffHoverContext);

    const handleMouseEnter = () => setHoveredId(diff.id, true);
    const handleMouseLeave = () => setHoveredId(diff.id, false);

    return (
        diff && (
            <Group
                key={diff.id} //{diff.id + (isSource ? '_source' : '_target')}
                onMouseOver={handleMouseEnter}
                onMouseOut={handleMouseLeave}
            >
                {diff.box?.map((box, idx) => (
                    <Rect
                        key={idx}// key={diff.id + idx + (isSource ? '_source' : '_target')}
                        x={box.x1 - 2}
                        y={box.y1 - 0.05 * (box.y2 - box.y1)}
                        width={box.x2 - box.x1 + 4}
                        height={box.y2 - box.y1 + 0.1 * (box.y2 - box.y1)}
                        fill={isSource ? ((diff.type === 'delete') ? 'red' : 'blue'): ((diff.type === 'insert') ? 'green' : 'blue')}
                        opacity={hoveredIds.includes(diff.id) ? 0.6 : 0.3}
                    />
                ))}
            </Group>
        )
    );
});


const Tooltip = ({ visible, position, content }: any) => (
    visible && (
      <Group>
        <Text
          text={content}
          x={position.x}
          y={position.y}
          fill='black'
        />
        <Rect
          x={position.x}
          y={position.y}
          width={content.length * 7}
          height={15}
          fill='blue'
          opacity={0.3}
        />
      </Group>
    )
  )

interface Bbox {
    x1: number;
    y1: number;
    x2: number;
    y2: number
}


interface DiffElement {
    id: string;
    box: Bbox[];
    type: EntityType;
}

interface Dlo {
    id: string;
    box: Bbox;
    label: string;
    score: number;
    page: number;
    type: EntityType;
    isSource: boolean;
    diff: DiffElement[];
}


interface DloProps extends Dlo{
    handleSelectDlo: (id: string) => void;
}

const DloComponent: React.FC<DloProps> = (props) => {
    const { 
        id, 
        box, 
        label, 
        score, 
        page, 
        type, 
        isSource, 
        diff,
        handleSelectDlo
    } = props;
    
    const { hoveredIds, setHoveredId } = useContext(DloHoverContext);

    const [tooltipVisible, setTooltipVisible] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
    const [tooltipContent, setTooltipContent] = useState('');

    const handleMouseEnter = (e: any) => {
        setTooltipVisible(true);
        setTooltipPosition(e.target.getStage().getPointerPosition());
        setTooltipContent(`${label} (${score.toFixed(2)})`);
        setHoveredId(id, true);
    };
    const handleMouseLeave = () => {
        setTooltipVisible(false);
        setHoveredId(id, false);
    };
    const handleOnClick = () => handleSelectDlo(id);

    const color = isSource ? ((type === 'delete') ? 'red' : 'blue'): ((type === 'insert') ? 'green' : 'blue')

    return (
        <>
        <Group
            key={id} //{id + (isSource ? '_source' : '_target')}
            id={id}
            onMouseOver={handleMouseEnter}
            onMouseOut={handleMouseLeave}
            onClick={handleOnClick}
        >
            <Rect
                key={-1}
                x={box.x1}
                y={box.y1}
                width={box.x2 - box.x1}
                height={box.y2 - box.y1}
                stroke= {isSource ? ((type === 'delete') ? 'red' : 'blue'): ((type === 'insert') ? 'green' : 'blue')}
                strokeWidth={hoveredIds.includes(id) ? 3 : 1}
                opacity={hoveredIds.includes(id) ? 0.6 : 0.3}
            />
            {diff && 
                diff?.map((annots, ind) => {
                            return (
                                <DrawRects
                                    key={ind} //{annots.id + (isSource ? '_source' : '_target')}
                                    diff={annots} 
                                    index={annots.id} 
                                    isSource={isSource} 
                                />
                            );
                        })}
        </Group>
        <Tooltip
          visible={tooltipVisible}
          position={tooltipPosition}
          content={tooltipContent}
        />
        </>
        
    );
};


const Colors: { [key: string]: string } = {
    'insert': 'green',
    'delete': 'red',
    'replace': 'blue',
}


function base64ToBytes(base64String: string): Uint8Array {
  const binaryString = window.atob(base64String);
  const byteArray = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    byteArray[i] = binaryString.charCodeAt(i);
  }

  return byteArray;
}


interface DocProps {
    data?: {data: string},
    pageNumber?: number,
    dlos?: Dlo[],
    isSource: boolean,
    handleSelectDlo: (id: string) => void,
}

const Doc: React.FC<DocProps> = memo((props: DocProps) => {
    const [numPages, setNumPages] = useState<number>(0);
    const [pageWidth, setPageWidth] = useState<number>(0);
    const [pageHeight, setPageHeight] = useState<number>(0);

    const {
        data,
        pageNumber,
        dlos,
        isSource,
        handleSelectDlo,
    } = props;

    const pageDlos = dlos ? dlos : [];

    return (
        <div style={{ position: "relative" }}>
            {data && data.data ?
                <Document
                    file={{data: base64ToBytes(data.data)}}
                    onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                >
                    {pageNumber ? <Page
                        pageNumber={pageNumber + 1}
                        renderTextLayer={false}
                        renderAnnotationLayer={false}
                        onLoadSuccess={({ width, height }) =>
                            {setPageWidth(width); setPageHeight(height)}
                        }
                    /> : null}
                </Document> : null}
            <Stage
                width={pageWidth}
                height={pageHeight}
                style={{ zIndex: 2, position: "absolute", top: 0, left: 0 }}
            >
                <Layer>
                    {pageDlos &&
                        pageDlos?.map((dlo) => dlo && dlo.id ? <DloComponent
                                key={dlo.id + (isSource ? '_source' : '_target')}
                                id={dlo.id}
                                box={dlo.box}
                                label={dlo.label}
                                score={dlo.score}
                                page={dlo.page}
                                type={dlo.type}
                                isSource={isSource}
                                diff={dlo.diff}
                                handleSelectDlo={handleSelectDlo}
                            /> : null
                        )}
                </Layer>
            </Stage>
            <p>
                Page {pageNumber} of {numPages}
            </p>
        </div>
    );
}, (prevProps, nextProps) => {
    return prevProps.pageNumber === nextProps.pageNumber;
});


interface DloHoverContextType {
    hoveredIds: string[];
    setHoveredId: (id: string, isHovered: boolean) => void;
}

const DloHoverContext = createContext<DloHoverContextType>({ hoveredIds: [], setHoveredId: () => {} });

interface DloHoverProviderProps {
  children: ReactNode;
}

const DloHoverProvider: React.FC<DloHoverProviderProps> = ({ children }) => {
  const [hoveredIds, setHoveredIds] = useState<string[]>([]);  
  
  const setHoveredId = (id: string, isHovered: boolean) => {
    if (isHovered) {
      setHoveredIds(prev => [...prev, id]);
    } else {
      setHoveredIds(prev => prev.filter(i => i !== id));
    }
  };

  return (
    <DloHoverContext.Provider value={{ hoveredIds, setHoveredId }}>
      {children}
    </DloHoverContext.Provider>
  );
};


interface DiffHoverContextType {
    hoveredIds: string[];
    setHoveredId: (id: string, isHovered: boolean) => void;
}

const DiffHoverContext = createContext<DiffHoverContextType>({ hoveredIds: [], setHoveredId: () => {} });

interface DiffHoverProviderProps {
  children: ReactNode;
}

const DiffHoverProvider: React.FC<DiffHoverProviderProps> = ({ children }) => {
  const [hoveredIds, setHoveredIds] = useState<string[]>([]);  
  
  const setHoveredId = (id: string, isHovered: boolean) => {
    if (isHovered) {
      setHoveredIds(prev => [...prev, id]);
    } else {
      setHoveredIds(prev => prev.filter(i => i !== id));
    }
  };

  return (
    <DiffHoverContext.Provider value={{ hoveredIds, setHoveredId }}>
      {children}
    </DiffHoverContext.Provider>
  );
};


interface PDFDiffState {
    onHoverDloId?: string
    onHoverDiffId?: string
    selectedDloId?: string
}

interface PDFDiffProps {
    args: {
      old_data: any,
      new_data: any,
      diff: any,
    }
  }
  
const PDFDiff: React.FC<PDFDiffProps> = ({ args }) => {

  const [state, setState] = useState<PDFDiffState>({
    onHoverDloId: "",
    onHoverDiffId: "",
    selectedDloId: "",
  });

  const { old_data, new_data, diff } = args;
  
  const { sourceIdPageMap, targetIdPageMap, entities, source, target } = useMemo(() => {
    const entities = transformEntities(diff);
    const source = entities[0];
    const target = entities[1];
    const sourceIdPageMap = getDloPageMap(source);
    const targetIdPageMap = getDloPageMap(target);

    return { sourceIdPageMap, targetIdPageMap, entities, source, target };
  }, [diff]);
  
  const { sourcePage, targetPage, sourcePageDlos, targetPageDlos } = useMemo(() => {
    const sourcePage = state.selectedDloId ? sourceIdPageMap.get(state.selectedDloId) : source[0].page;
    const targetPage = state.selectedDloId ? targetIdPageMap.get(state.selectedDloId) : target[0].page;

    const sourcePageDlos = sourcePage ? getDloByPage(source).get(sourcePage) : [];
    const targetPageDlos = targetPage ? getDloByPage(target).get(targetPage) : [];

    return { sourcePage, targetPage, sourcePageDlos, targetPageDlos };
  }, [state.selectedDloId, source, target, sourceIdPageMap, targetIdPageMap]);

  const handleSelectDlo = useCallback((id: string) => {
    console.log("select")
    setState(prevState => ({...prevState, selectedDloId: id }));
    Streamlit.setComponentValue(id);
  }, []);

  return (
    <DloHoverProvider>
      <DiffHoverProvider>
      <div style={{
          display: "grid",
          gridAutoColumns: "1500px",
          gridAutoRows: "1500px",
          gridTemplateColumns: "repeat(2, 1fr)",
          gridGap: 20
      }}>
        
          <Doc
              data={old_data}
              dlos={sourcePageDlos}
              pageNumber={sourcePage}
              isSource={true}
              handleSelectDlo={handleSelectDlo}
          />
          <Doc
              data={new_data}
              dlos={targetPageDlos}
              pageNumber={targetPage}
              isSource={false}
              handleSelectDlo={handleSelectDlo}
          />
      </div>
      </DiffHoverProvider>
    </DloHoverProvider>
  );
}


class PDFDiffStreamlitWrapper extends StreamlitComponentBase<PDFDiffState> {
    public render = (): ReactNode => {
        return (
            <PDFDiff
                args={{
                    old_data: this.props.args["old_data"],
                    new_data: this.props.args["new_data"],
                    diff: this.props.args["diff"],
                }}
            />
        );
    };
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
export default withStreamlitConnection(PDFDiffStreamlitWrapper)