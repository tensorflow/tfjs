/**
 * @license
 * Copyright 2024 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * @license
 * Copyright 2024 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import{loadGraphModel as t}from"@tensorflow/tfjs-converter";import{Tensor as o,browser as r,util as e,tidy as s,expandDims as n,image as c,sub as a,div as i,cast as l,dispose as p}from"@tensorflow/tfjs-core";
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function u(t){return t instanceof o?t:r.fromPixels(t)}async function f(t){const o=t.lastIndexOf("/"),r=`${o>=0?t.slice(0,o+1):""}dict.txt`,s=await e.fetch(r);return(await s.text()).trim().split("\n")}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const h=[224,224];class d{constructor(t,o){this.graphModel=t,this.dictionary=o}async classify(t,o){o=function(t){null==(t=t||{}).centerCrop&&(t.centerCrop=!0);return t}(o);const r=s((()=>{const r=this.preprocess(t,o);return this.graphModel.predict(r)})),e=await r.data();r.dispose();return Array.from(e).map(((t,o)=>({label:this.dictionary[o],prob:t})))}preprocess(t,o){const r=u(t),e=o.centerCrop?function(t){return s((()=>{const[o,r]=t.shape.slice(0,2);let e=0,s=0;o>r?e=(o-r)/2:s=(r-o)/2;const a=Math.min(r,o),i=[[e/o,s/r,(e+a)/o,(s+a)/r]],p=[0];return c.cropAndResize(n(l(t,"float32")),i,p,h)}))}
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */(r):n(c.resizeBilinear(r,h));return a(i(e,127.5),1)}}async function m(o){const[r,e]=await Promise.all([t(o),f(o)]);return new d(r,e)}const w=["Postprocessor/convert_scores","Postprocessor/Decode/transpose_1"];class y{constructor(t,o){this.graphModel=t,this.dictionary=o}async detect(t,o){o=function(t){null==(t=t||{}).topk&&(t.topk=20);null==t.iou&&(t.iou=.5);null==t.score&&(t.score=.5);return t}(o);const r=s((()=>this.preprocess(t,o))),[e,n]=[r.shape[1],r.shape[2]],a={};a.ToFloat=r;const[i,l]=await this.graphModel.executeAsync(a,w),[,u,f]=i.shape,[h,d]=await Promise.all([i.data(),l.data()]),{boxScores:m,boxLabels:y}=function(t,o,r){const e=[],s=[];for(let n=0;n<o;n++){let o=Number.MIN_VALUE,c=-1;for(let e=0;e<r;e++){const s=n*r+e;t[s]>o&&(o=t[s],c=e)}e[n]=o,s[n]=c}return{boxScores:e,boxLabels:s}}(h,u,f),x=await c.nonMaxSuppressionAsync(l,m,o.topk,o.iou,o.score),b=await x.data();p([r,i,l,x]);const M=function(t,o,r,e,s,n,c){const a=[],i=4;for(let l=0;l<n.length;l++){const p=n[l],[u,f,h,d]=Array.from(r.slice(p*i,p*i+i));a.push({box:{left:f*t,top:u*o,width:(d-f)*t,height:(h-u)*o},label:c[s[p]],score:e[p]})}return a}
/** @license See the LICENSE file. */(n,e,d,m,y,b,this.dictionary);return M}preprocess(t,o){return l(n(u(t)),"float32")}}async function x(o){const[r,e]=await Promise.all([t(o),f(o)]);return new y(r,e)}const b="1.2.0";export{d as ImageClassificationModel,y as ObjectDetectionModel,m as loadImageClassification,x as loadObjectDetection,b as version};
//# sourceMappingURL=tf-automl.esm.js.map
