(function(root){
    'use strict';
    // Only object URLs in page memory. No files, browser storage, or extra uploads.
    function createStore(urls,comparable){
        let current=null,previous=null;
        const key=row=>row?JSON.stringify(row):null;
        function release(item){if(item)urls.revokeObjectURL(item.url);}
        function clear(){release(previous);release(current);previous=current=null;}
        function commit(blob,row,priorRow){
            if(!blob){clear();return;}
            let url;
            try{url=urls.createObjectURL(blob);}catch(_){clear();return;}
            const keep=current && current.key===key(priorRow) && comparable(row.diagnosis,current.diagnosis);
            release(previous);
            if(!keep)release(current);
            previous=keep?current:null;
            current={url,key:key(row),diagnosis:row.diagnosis};
        }
        return {commit,clear,snapshot:()=>({previous:previous?.url||null,current:current?.url||null})};
    }
    root.AudioComparison={createStore};
    if(typeof module!=='undefined')module.exports=root.AudioComparison;
})(globalThis);
